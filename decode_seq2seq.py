"""BERT finetuning runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import logging
import math
import os
import pickle
import random
import json
from time import sleep

import numpy as np
import torch
from nltk.translate.bleu_score import sentence_bleu
from tqdm import tqdm
from transformers import \
    BertTokenizer, RobertaTokenizer
from transformers.tokenization_bert import whitespace_tokenize

import s2s_ft.s2s_loader as seq2seq_loader
from s2s_ft.modeling_decoding import LayoutlmForSeq2SeqDecoder, BertConfig
from s2s_ft.tokenization_minilm import MinilmTokenizer
from s2s_ft.tokenization_unilm import UnilmTokenizer
from s2s_ft.utils import load_and_cache_layoutlm_examples, convert_src_layout_inputs_to_tokens, \
    get_tokens_from_src_and_index, convert_tgt_layout_inputs_to_tokens

TOKENIZER_CLASSES = {
    'bert': BertTokenizer,
    'minilm': MinilmTokenizer,
    'roberta': RobertaTokenizer,
    'unilm': UnilmTokenizer,
    'layoutlm': BertTokenizer,
}


class WhitespaceTokenizer(object):
    def tokenize(self, text):
        return whitespace_tokenize(text)


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def detokenize(tk_list):
    r_list = []
    for tk in tk_list:
        if tk.startswith('##') and len(r_list) > 0:
            r_list[-1] = r_list[-1] + tk[2:]
        else:
            r_list.append(tk)
    return r_list


def ascii_print(text):
    text = text.encode("ascii", "ignore")
    print(text)


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="Model type selected in the list: " + ", ".join(TOKENIZER_CLASSES.keys()))
    parser.add_argument("--model_path", default=None, type=str, required=True,
                        help="Path to the model checkpoint.")
    parser.add_argument("--config_path", default=None, type=str,
                        help="Path to config.json for the model.")

    parser.add_argument("--sentence_shuffle_rate", default=0, type=float)
    parser.add_argument("--layoutlm_only_layout", action='store_true')

    # tokenizer_name
    parser.add_argument("--tokenizer_name", default=None, type=str, required=True,
                        help="tokenizer name")
    parser.add_argument("--max_seq_length", default=512, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")

    # decoding parameters
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--amp', action='store_true',
                        help="Whether to use amp for fp16")
    parser.add_argument("--input_file", type=str, help="Input file")
    parser.add_argument("--input_folder", type=str, help="Input folder")
    parser.add_argument("--cached_feature_file", type=str)
    parser.add_argument('--subset', type=int, default=0,
                        help="Decode a subset of the input dataset.")
    parser.add_argument("--output_file", type=str, help="output file")
    parser.add_argument("--split", type=str, default="",
                        help="Data split (train/val/test).")
    parser.add_argument('--tokenized_input', action='store_true',
                        help="Whether the input is tokenized.")
    parser.add_argument('--seed', type=int, default=123,
                        help="random seed for initialization")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument('--batch_size', type=int, default=4,
                        help="Batch size for decoding.")
    parser.add_argument('--beam_size', type=int, default=1,
                        help="Beam size for searching")
    parser.add_argument('--length_penalty', type=float, default=0,
                        help="Length penalty for beam search")

    parser.add_argument('--forbid_duplicate_ngrams', action='store_true')
    parser.add_argument('--forbid_ignore_word', type=str, default=None,
                        help="Forbid the word during forbid_duplicate_ngrams")
    parser.add_argument("--min_len", default=1, type=int)
    parser.add_argument('--need_score_traces', action='store_true')
    parser.add_argument('--ngram_size', type=int, default=3)
    parser.add_argument('--mode', default="s2s",
                        choices=["s2s", "l2r", "both"])
    parser.add_argument('--max_tgt_length', type=int, default=128,
                        help="maximum length of target sequence")
    parser.add_argument('--s2s_special_token', action='store_true',
                        help="New special tokens ([S2S_SEP]/[S2S_CLS]) of S2S.")
    parser.add_argument('--s2s_add_segment', action='store_true',
                        help="Additional segmental for the encoder of S2S.")
    parser.add_argument('--s2s_share_segment', action='store_true',
                        help="Sharing segment embeddings for the encoder of S2S (used with --s2s_add_segment).")
    parser.add_argument('--pos_shift', action='store_true',
                        help="Using position shift for fine-tuning.")
    parser.add_argument("--cache_dir", default=None, type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")

    args = parser.parse_args()

    model_path = args.model_path
    assert os.path.exists(model_path), 'model_path ' + model_path + ' not exists!'

    if args.need_score_traces and args.beam_size <= 1:
        raise ValueError(
            "Score trace is only available for beam search with beam size > 1.")
    if args.max_tgt_length >= args.max_seq_length - 2:
        raise ValueError("Maximum tgt length exceeds max seq length - 2.")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()

    if args.seed > 0:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if n_gpu > 0:
            torch.cuda.manual_seed_all(args.seed)
    else:
        random_seed = random.randint(0, 10000)
        logger.info("Set random seed as: {}".format(random_seed))
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        if n_gpu > 0:
            torch.cuda.manual_seed_all(args.seed)

    tokenizer = TOKENIZER_CLASSES[args.model_type].from_pretrained(
        args.tokenizer_name, do_lower_case=args.do_lower_case,
        cache_dir=args.cache_dir if args.cache_dir else None,
        max_len=args.max_seq_length
    )

    if args.model_type == "roberta":
        vocab = tokenizer.encoder
    else:
        vocab = tokenizer.vocab

    # NOTE: tokenizer cannot setattr, so move this to the initialization step
    # tokenizer.max_len = args.max_seq_length

    config_file = args.config_path if args.config_path else os.path.join(args.model_path, "config.json")
    logger.info("Read decoding config from: %s" % config_file)
    config = BertConfig.from_json_file(config_file,
                                       # base_model_type=args.model_type
                                       layoutlm_only_layout_flag=args.layoutlm_only_layout
                                       )

    bi_uni_pipeline = []
    bi_uni_pipeline.append(seq2seq_loader.Preprocess4Seq2seqDecoder(
        list(vocab.keys()), tokenizer.convert_tokens_to_ids, args.max_seq_length,
        max_tgt_length=args.max_tgt_length, pos_shift=args.pos_shift,
        source_type_id=config.source_type_id, target_type_id=config.target_type_id,
        cls_token=tokenizer.cls_token, sep_token=tokenizer.sep_token, pad_token=tokenizer.pad_token,
        layout_flag=args.model_type == 'layoutlm'
    ))

    mask_word_id, eos_word_ids, sos_word_id = tokenizer.convert_tokens_to_ids(
        [tokenizer.mask_token, tokenizer.sep_token, tokenizer.sep_token])
    forbid_ignore_set = None
    if args.forbid_ignore_word:
        w_list = []
        for w in args.forbid_ignore_word.split('|'):
            if w.startswith('[') and w.endswith(']'):
                w_list.append(w.upper())
            else:
                w_list.append(w)
        forbid_ignore_set = set(tokenizer.convert_tokens_to_ids(w_list))
    found_checkpoint_flag = False
    for model_recover_path in [args.model_path.strip()]:
        logger.info("***** Recover model: %s *****", model_recover_path)
        found_checkpoint_flag = True
        model = LayoutlmForSeq2SeqDecoder.from_pretrained(
            model_recover_path, config=config, mask_word_id=mask_word_id, search_beam_size=args.beam_size,
            length_penalty=args.length_penalty, eos_id=eos_word_ids, sos_id=sos_word_id,
            forbid_duplicate_ngrams=args.forbid_duplicate_ngrams, forbid_ignore_set=forbid_ignore_set,
            ngram_size=args.ngram_size, min_len=args.min_len, mode=args.mode,
            max_position_embeddings=args.max_seq_length, pos_shift=args.pos_shift,
        )

        if args.fp16:
            model.half()
        model.to(device)
        if n_gpu > 1:
            model = torch.nn.DataParallel(model)

        torch.cuda.empty_cache()
        model.eval()
        next_i = 0
        max_src_length = args.max_seq_length - 2 - args.max_tgt_length
        max_tgt_length = args.max_tgt_length

        example_path = args.input_file if args.input_file else args.input_folder

        to_pred = load_and_cache_layoutlm_examples(
            example_path, tokenizer, local_rank=-1,
            cached_features_file=args.cached_feature_file, shuffle=False, layout_flag=args.model_type == 'layoutlm',
            src_shuffle_rate=args.sentence_shuffle_rate
        )

        input_lines = convert_src_layout_inputs_to_tokens(to_pred, tokenizer.convert_ids_to_tokens, max_src_length,
                                                          layout_flag=args.model_type == 'layoutlm')
        target_lines = convert_tgt_layout_inputs_to_tokens(to_pred, tokenizer.convert_ids_to_tokens, max_tgt_length,
                                                           layout_flag=args.model_type == 'layoutlm')

        target_geo_scores = [x['bleu'] for x in to_pred]

        if args.subset > 0:
            logger.info("Decoding subset: %d", args.subset)
            input_lines = input_lines[:args.subset]

        # NOTE: add the sequence index through enumerate
        input_lines = sorted(list(enumerate(input_lines)), key=lambda x: -len(x[1]))


        score_trace_list = [None] * len(input_lines)
        total_batch = math.ceil(len(input_lines) / max(1, args.batch_size))

        fn_out = args.output_file
        fout = open(fn_out, "w", encoding="utf-8")

        with tqdm(total=total_batch) as pbar:
            batch_count = 0
            first_batch = True

            while first_batch or (next_i + args.batch_size <= len(input_lines)):
            # while next_i < len(input_lines):
                _chunk = input_lines[next_i:next_i + args.batch_size]


                buf_id = [x[0] for x in _chunk]
                buf = [x[1] for x in _chunk]

            

                next_i += args.batch_size
                batch_count += 1
                max_a_len = max([len(x) for x in buf])
                instances = []
                for instance in [(x, max_a_len) for x in buf]:
                    for proc in bi_uni_pipeline:
                        instances.append(proc(instance))
                with torch.no_grad():
                    batch = seq2seq_loader.batch_list_to_batch_tensors(instances)
                    batch = [t.to(device) if t is not None else None for t in batch]

                    input_ids, token_type_ids, position_ids, input_mask, mask_qkv, task_idx = batch
                    traces = model(input_ids, token_type_ids, position_ids, input_mask, task_idx=task_idx, mask_qkv=mask_qkv)

                    if args.beam_size > 1:
                        traces = {k: v.tolist() for k, v in traces.items()}
                        output_ids = traces['pred_seq']
                    else:
                        output_ids = traces.tolist()

                    for i in range(len(buf)):

                        w_ids = output_ids[i]
                        """
                        src = buf[i] = [['va', 64.0, 2.0, 70.0, 3.0], ['##sh', 64.0, 2.0, 70.0, 3.0], ['##na', 64.0, 2.0, 70.0, 3.0], ['ram', 71.0, 2.0, 78.0, 3.0], ['##nath', 71.0, 2.0, 78.0, 3.0], ['op', 64.0, 5.0, 75.0, 6.0], ['##lei', 64.0, 5.0, 75.0, 6.0], ['##ding', 64.0, 5.0, 75.0, 6.0], ['##en', 64.0, 5.0, 75.0, 6.0], ['front', 64.0, 8.0, 75.0, 9.0], ['##off', 64.0, 8.0, 75.0, 9.0], ['##ice', 64.0, 8.0, 75.0, 9.0], ['management', 76.0, 8.0, 86.0, 9.0], ['mb', 87.0, 8.0, 90.0, 9.0], ['##o', 87.0, 8.0, 90.0, 9.0], ['4', 91.0, 8.0, 92.0, 9.0], ['(', 93.0, 8.0, 102.0, 9.0], ['be', 93.0, 8.0, 102.0, 9.0], ['##ha', 93.0, 8.0, 102.0, 9.0], ['##ald', 93.0, 8.0, 102.0, 9.0], [')', 93.0, 8.0, 102.0, 9.0], ['2010', 169.0, 8.0, 173.0, 9.0], ['-', 174.0, 8.0, 175.0, 9.0], ['2011', 176.0, 8.0, 180.0, 9.0], ['no', 64.0, 9.0, 76.0, 10.0], ['##ord', 64.0, 9.0, 76.0, 10.0], ['##er', 64.0, 9.0, 76.0, 10.0], ['##poo', 64.0, 9.0, 76.0, 10.0], ['##rt', 64.0, 9.0, 76.0, 10.0], ['college', 77.0, 9.0, 85.0, 10.0], [',', 77.0, 9.0, 85.0, 10.0], ['groningen', 86.0, 9.0, 95.0, 10.0], ['front', 64.0, 11.0, 75.0, 12.0], ['##off', 64.0, 11.0, 75.0, 12.0], ['##ice', 64.0, 11.0, 75.0, 12.0], ['(', 76.0, 11.0, 85.0, 12.0], ['be', 76.0, 11.0, 85.0, 12.0], ['##ha', 76.0, 11.0, 85.0, 12.0], ['##ald', 76.0, 11.0, 85.0, 12.0], [')', 76.0, 11.0, 85.0, 12.0], ['2008', 169.0, 11.0, 173.0, 12.0], ['-', 174.0, 11.0, 175.0, 12.0], ['2010', 176.0, 11.0, 180.0, 12.0], ['al', 64.0, 12.0, 70.0, 13.0], ['##bed', 64.0, 12.0, 70.0, 13.0], ['##a', 64.0, 12.0, 70.0, 13.0], ['college', 71.0, 12.0, 79.0, 13.0], [',', 71.0, 12.0, 79.0, 13.0], ['rotterdam', 80.0, 12.0, 89.0, 13.0], ['personal', 0.0, 13.0, 10.0, 14.0], ['##ia', 0.0, 13.0, 10.0, 14.0], ['va', 8.0, 15.0, 14.0, 16.0], ['##sh', 8.0, 15.0, 14.0, 16.0], ['##na', 8.0, 15.0, 14.0, 16.0], ['ram', 15.0, 15.0, 22.0, 16.0], ['##nath', 15.0, 15.0, 22.0, 16.0], ['cu', 64.0, 16.0, 73.0, 17.0], ['##rs', 64.0, 16.0, 73.0, 17.0], ['##uss', 64.0, 16.0, 73.0, 17.0], ['##en', 64.0, 16.0, 73.0, 17.0], ['ve', 64.0, 19.0, 71.0, 20.0], ['##rk', 64.0, 19.0, 71.0, 20.0], ['##oop', 64.0, 19.0, 71.0, 20.0], ['advise', 72.0, 19.0, 80.0, 20.0], ['##ur', 72.0, 19.0, 80.0, 20.0], ['(', 81.0, 19.0, 90.0, 20.0], ['be', 81.0, 19.0, 90.0, 20.0], ['##ha', 81.0, 19.0, 90.0, 20.0], ['##ald', 81.0, 19.0, 90.0, 20.0], [')', 81.0, 19.0, 90.0, 20.0], ['2013', 181.0, 19.0, 185.0, 20.0], ['w', 64.0, 21.0, 67.0, 22.0], ['##ft', 64.0, 21.0, 67.0, 22.0], ['z', 68.0, 21.0, 85.0, 22.0], ['##org', 68.0, 21.0, 85.0, 22.0], ['##ver', 68.0, 21.0, 85.0, 22.0], ['##zek', 68.0, 21.0, 85.0, 22.0], ['##ering', 68.0, 21.0, 85.0, 22.0], ['##en', 68.0, 21.0, 85.0, 22.0], ['(', 86.0, 21.0, 95.0, 22.0], ['be', 86.0, 21.0, 95.0, 22.0], ['##ha', 86.0, 21.0, 95.0, 22.0], ['##ald', 86.0, 21.0, 95.0, 22.0], [')', 86.0, 21.0, 95.0, 22.0], ['2017', 181.0, 21.0, 185.0, 22.0], ['de', 8.0, 24.0, 10.0, 25.0], ['graf', 11.0, 24.0, 16.0, 25.0], ['##t', 11.0, 24.0, 16.0, 25.0], ['12', 17.0, 24.0, 19.0, 25.0], ['we', 64.0, 24.0, 76.0, 25.0], ['##rke', 64.0, 24.0, 76.0, 25.0], ['##rva', 64.0, 24.0, 76.0, 25.0], ['##ring', 64.0, 24.0, 76.0, 25.0], ['92', 8.0, 25.0, 12.0, 26.0], ['##01', 8.0, 25.0, 12.0, 26.0], ['x', 13.0, 25.0, 15.0, 26.0], ['##s', 13.0, 25.0, 15.0, 26.0], ['dr', 16.0, 25.0, 24.0, 26.0], ['##ach', 16.0, 25.0, 24.0, 26.0], ['##ten', 16.0, 25.0, 24.0, 26.0], ['e', 64.0, 27.0, 72.0, 28.0], ['##igen', 64.0, 27.0, 72.0, 28.0], ['##aar', 64.0, 27.0, 72.0, 28.0], ['2021', 162.0, 27.0, 166.0, 28.0], ['-', 167.0, 27.0, 168.0, 28.0], ['jun', 169.0, 27.0, 173.0, 28.0], ['.', 169.0, 27.0, 173.0, 28.0], ['2021', 162.0, 27.0, 166.0, 28.0], ['2', 8.0, 28.0, 9.0, 29.0], ['jan', 10.0, 28.0, 17.0, 29.0], ['##ua', 10.0, 28.0, 17.0, 29.0], ['##ri', 10.0, 28.0, 17.0, 29.0], ['1990', 18.0, 28.0, 22.0, 29.0], ['mal', 64.0, 29.0, 70.0, 30.0], ['##hoe', 64.0, 29.0, 70.0, 30.0], ['food', 71.0, 29.0, 75.0, 30.0], ['&', 76.0, 29.0, 77.0, 30.0], ['service', 78.0, 29.0, 86.0, 30.0], [',', 78.0, 29.0, 86.0, 30.0], ['dr', 87.0, 29.0, 95.0, 30.0], ['##ach', 87.0, 29.0, 95.0, 30.0], ['##ten', 87.0, 29.0, 95.0, 30.0], ['e', 64.0, 30.0, 69.0, 31.0], ['##igen', 64.0, 30.0, 69.0, 31.0], ['on', 70.0, 30.0, 81.0, 31.0], ['##dern', 70.0, 30.0, 81.0, 31.0], ['##emi', 70.0, 30.0, 81.0, 31.0], ['##ng', 70.0, 30.0, 81.0, 31.0], ['in', 78.0, 30.0, 80.0, 31.0], ['catering', 85.0, 30.0, 94.0, 31.0], ['.', 85.0, 30.0, 94.0, 31.0], ['rotterdam', 8.0, 31.0, 17.0, 32.0], ['ink', 64.0, 32.0, 78.0, 33.0], ['##oop', 64.0, 32.0, 78.0, 33.0], ['/', 64.0, 32.0, 78.0, 33.0], ['ve', 64.0, 32.0, 78.0, 33.0], ['##rk', 64.0, 32.0, 78.0, 33.0], ['##oop', 64.0, 32.0, 78.0, 33.0], ['product', 79.0, 32.0, 89.0, 33.0], ['##en', 79.0, 32.0, 89.0, 33.0], [',', 79.0, 32.0, 89.0, 33.0], ['sur', 90.0, 32.0, 99.0, 33.0], ['##ina', 90.0, 32.0, 99.0, 33.0], ['##ams', 90.0, 32.0, 99.0, 33.0], ['et', 100.0, 32.0, 104.0, 33.0], ['##en', 100.0, 32.0, 104.0, 33.0], ['vo', 105.0, 32.0, 118.0, 33.0], ['##or', 105.0, 32.0, 118.0, 33.0], ['##ber', 105.0, 32.0, 118.0, 33.0], ['##ei', 105.0, 32.0, 118.0, 33.0], ['##den', 105.0, 32.0, 118.0, 33.0], [',', 105.0, 32.0, 118.0, 33.0], ['bo', 119.0, 32.0, 131.0, 33.0], ['##ek', 119.0, 32.0, 131.0, 33.0], ['##ho', 119.0, 32.0, 131.0, 33.0], ['##uding', 119.0, 32.0, 131.0, 33.0], ['.', 119.0, 32.0, 131.0, 33.0], ['men', 64.0, 34.0, 70.0, 35.0], ['##zi', 64.0, 34.0, 70.0, 35.0], ['##s', 64.0, 34.0, 70.0, 35.0], ['z', 71.0, 34.0, 86.0, 35.0], ['##org', 71.0, 34.0, 86.0, 35.0], ['##ver', 71.0, 34.0, 86.0, 35.0], ['##zek', 71.0, 34.0, 86.0, 35.0], ['##ering', 71.0, 34.0, 86.0, 35.0], ['2019', 169.0, 34.0, 173.0, 35.0], ['-', 174.0, 34.0, 175.0, 35.0], ['2020', 176.0, 34.0, 180.0, 35.0], ['b', 8.0, 35.0, 9.0, 36.0], ['ce', 64.0, 36.0, 72.0, 37.0], ['##nd', 64.0, 36.0, 72.0, 37.0], ['##ris', 64.0, 36.0, 72.0, 37.0], [',', 64.0, 36.0, 72.0, 37.0], ['groningen', 73.0, 36.0, 82.0, 37.0], ['klan', 64.0, 37.0, 78.0, 38.0], ['##tens', 64.0, 37.0, 78.0, 38.0], ['##er', 64.0, 37.0, 78.0, 38.0], ['##vic', 64.0, 37.0, 78.0, 38.0], ['##e', 64.0, 37.0, 78.0, 38.0], ['med', 79.0, 37.0, 90.0, 38.0], ['##ew', 79.0, 37.0, 90.0, 38.0], ['##er', 79.0, 37.0, 90.0, 38.0], ['##ker', 79.0, 37.0, 90.0, 38.0], [',', 79.0, 37.0, 90.0, 38.0], ['diverse', 91.0, 37.0, 98.0, 38.0], ['vr', 99.0, 37.0, 105.0, 38.0], ['##age', 99.0, 37.0, 105.0, 38.0], ['##n', 99.0, 37.0, 105.0, 38.0], ['bean', 106.0, 37.0, 118.0, 38.0], ['##t', 106.0, 37.0, 118.0, 38.0], ['##wo', 106.0, 37.0, 118.0, 38.0], ['##ord', 106.0, 37.0, 118.0, 38.0], ['##en', 106.0, 37.0, 118.0, 38.0], ['van', 119.0, 37.0, 122.0, 38.0], ['klan', 123.0, 37.0, 130.0, 38.0], ['##ten', 123.0, 37.0, 130.0, 38.0], ['met', 131.0, 37.0, 134.0, 38.0], ['vr', 8.0, 38.0, 13.0, 39.0], ['##ou', 8.0, 38.0, 13.0, 39.0]]
                        w_ids = w_ids = [1, 2,
                            merged_string = "" 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 22, 23, 24, 41, 42, 43, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 44, 45, 46, 47, 48, 49, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 86, 87, 88, 89, 94, 95, 96, 97, 98, 99, 100, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 90, 91, 92, 93, 101, 102, 103, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 104, 105, 106, 107, 108, 132, 169, 198, 199, 200, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187 

                        def get_tokens_from_src_and_index(src, index, modifier=None):
                            result = []
                            for i in index:
                                i = modifier(i)
                                i = min(i, len(src) - 1)
                                if isinstance(src[i], list):
                                    result.append(src[i][0])
                                else:
                                    result.append(src[i])
                            return result

                        output_buf = ['va', '##sh', '##na', 'ram', '##nath', 'op', '##lei', '##ding', '##en', 'front', '##off', '##ice', 'management', 'mb', '##o', '4', '(', 'be', '##ha', '##ald', ')', 'no', '##ord', ...]
                        """

                        def get_bboxes_from_src_and_index(src, index, modifier=None):
                            result = []
                            for i in index:
                                i = modifier(i)
                                i = min(i, len(src) - 1)
                                if isinstance(src[i], list):
                                    result.append(src[i][1:5])
                                else:
                                    result.append(src[i])
                            return result
                        
                        #print("buf[i] =", buf[i])


                        def merged_strings(buf):
                            
                            merged_strings = []
                            bboxes = []
                            merged_string = ""

                            for j in range(len(buf)):
                                
                                if j > 0 and buf[j][1:5] == buf[j-1][1:5]:
                                    s = buf[j][0]

                                    if s.startswith('##'):
                                        s = buf[j][0][2:]

                                    merged_string += s
                                    
                                else:
                                    if merged_string:
                                        bbox = buf[j-1][1:5]
                                        bboxes.append(bbox)
                                        merged_strings.append(merged_string)
                                    merged_string = buf[j][0]
                                
                            # Append the last merged_string and its bounding box if it exists
                            if merged_string:
                                bbox = buf[-1][1:5]
                                bboxes.append(bbox)
                                merged_strings.append(merged_string)

                            return merged_strings, bboxes
             
                        merged_strings, bboxes = merged_strings(buf[i])

                        #print("len(merged_strings)", len(merged_strings))
                        #print("merged_strings =", merged_strings)

                        #print("len(bboxes)", len(bboxes))
                        #print("bboxes =", bboxes)

                        #print("w_ids =", w_ids)
                        output_buf = get_tokens_from_src_and_index(src=buf[i], index=w_ids, modifier=lambda x: x-1)

                        output_bboxes= get_bboxes_from_src_and_index(src=buf[i], index=w_ids, modifier=lambda x: x-1)
                        cleaned_output_bboxes = []
                        output_tokens = []
                        #print("output_buf =", output_buf)

                        # Remove double bboxes
                        for k, t in enumerate(output_buf):
                            if t in (tokenizer.sep_token, tokenizer.pad_token):
                                break
                            output_tokens.append(t)
                            if not cleaned_output_bboxes or output_bboxes[k] != cleaned_output_bboxes[-1]:
                                cleaned_output_bboxes.append(output_bboxes[k])

                        output_bboxes = cleaned_output_bboxes
                        #print("buf_id[i] =", buf_id[i])
                        #print("target_lines[buf_id[i]] =", target_lines[buf_id[i]])

                        output_tokens = output_tokens[:len(target_lines[buf_id[i]])]
                        output_bboxes = output_bboxes[:len(output_bboxes)]
                        if args.model_type == "roberta":
                            output_sequence = tokenizer.convert_tokens_to_string(output_tokens)
                        else:
                            output_sequence = ' '.join(detokenize(output_tokens))
                        if '\n' in output_sequence:
                            output_sequence = " [X_SEP] ".join(output_sequence.split('\n'))

                        target = target_lines[buf_id[i]]
                        target = detokenize(target)

                        #result = output_sequence.split()

                        result = merged_strings

                        print("Result:", result, end="\n\n\n\n\n")

                        #print("len(result(bboxes)):", len(output_bboxes))
                        #print("Result bboxes:", output_bboxes)

                        with open("../../output_bboxes.json", mode="w") as f:
                            json.dump({"text":merged_strings, "bboxes" : bboxes}, f)

                        score = sentence_bleu([target], result)

                        geo_score = target_geo_scores[buf_id[i]]
                        target_sequence = ' '.join(target)

                        fout.write('{}\t{:.8f}\t{:.8f}\t{}\t{}\n'.format(buf_id[i], score, geo_score, output_sequence, target_sequence))

                        if first_batch or batch_count % 50 == 0:
                            logger.info("{}: BLEU={:.4f} GEO={:.4f} | {}"
                                        .format(buf_id[i], score, target_geo_scores[buf_id[i]], output_sequence))
                            
                        if args.need_score_traces:
                            score_trace_list[buf_id[i]] = {
                                'scores': traces['scores'][i], 'wids': traces['wids'][i], 'ptrs': traces['ptrs'][i]}
                    
                pbar.update(1)
                first_batch = False

        # for i, sequence in enumerate(output_sequence):
        #     print(f"Word: {sequence}, Position: {i}")

        outscore = open(fn_out, encoding='utf-8')
        bleu_score = geo_score = {}
        total_bleu = total_geo = 0.0
        for line in outscore.readlines():
            id, bleu, geo, out_seq, tgt_seq = line.split('\t')
            bleu_score[int(id)] = float(bleu)
            total_bleu += float(bleu)
            geo_score[int(id)] = float(geo)
            total_geo += float(geo)

        print("avg_bleu", round(100 * total_bleu / max(1, len(bleu_score)), 1))
        print("avg_geo", round(100 * total_geo / max(1, len(geo_score)), 1))
        # released model (layoutreader-base-readingbank): avg_bleu 98.2, avg_geo 69.7

        if args.need_score_traces:
            with open(fn_out + ".trace.pickle", "wb") as fout_trace:
                pickle.dump(
                    {"version": 0.0, "num_samples": len(input_lines)}, fout_trace)
                for x in score_trace_list:
                    pickle.dump(x, fout_trace)

    if not found_checkpoint_flag:
        logger.info("Not found the model checkpoint file!")


if __name__ == "__main__":
    main()
