# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for sequence classification on GLUE (Bert, XLM, XLNet, RoBERTa)."""

from __future__ import absolute_import, division, print_function

import argparse
import glob
import logging
import os
import random
import time
import datetime

import numpy as np
# import comet_ml as cm
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler

try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from tensorboardX import SummaryWriter

from tqdm import tqdm, trange

from transformers import (WEIGHTS_NAME, BertConfig,
                                  BertTokenizer,
                                  RobertaConfig,
                                  RobertaTokenizer,
                                  AlbertConfig,
                                  AlbertTokenizer,
                                  DistilBertConfig,
                                  DistilBertTokenizer)

from transformers.modeling_highway_bert import BertForSequenceClassification
from transformers.modeling_highway_roberta import RobertaForSequenceClassification
from transformers.modeling_highway_albert import AlbertForSequenceClassification
from transformers.modeling_highway_distilbert import DistilBertForSequenceClassification

from transformers import AdamW, get_linear_schedule_with_warmup

from transformers import glue_compute_metrics as compute_metrics
from transformers import glue_output_modes as output_modes
from transformers import glue_processors as processors
from transformers import glue_convert_examples_to_features as convert_examples_to_features

ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys()) for conf in (BertConfig,
                                                                                RobertaConfig, AlbertConfig, DistilBertConfig)), ())

MODEL_CLASSES = {
    'bert': (BertConfig, BertForSequenceClassification, BertTokenizer),
    'roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
    'albert': (AlbertConfig, AlbertForSequenceClassification, AlbertTokenizer),
    'distilbert': (DistilBertConfig, DistilBertForSequenceClassification, DistilBertTokenizer)
}

def get_args():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(
                            ALL_MODELS))
    parser.add_argument("--task_name", default=None, type=str, required=True,
                        help="The name of the task to train selected in the list: " + ", ".join(processors.keys()))
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--plot_data_dir", default="./plotting/", type=str, required=False,
                        help="The directory to store data for plotting figures.")

    ## Other parameters
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Rul evaluation during training at each logging step.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--eval_each_highway", action='store_true',
                        help="Set this flag to evaluate each highway.")
    parser.add_argument("--eval_highway", action='store_true',
                        help="Set this flag if it's evaluating highway models")

    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--early_exit_entropy", default=-1, type=float,
                        help="Entropy threshold for early exit.")
    parser.add_argument("--lte_th", default=None, type=str,
                        help="Learning to exit threshold. Example:"
                             "'0.2' or '0.3,4;0.2,8'")
    parser.add_argument("--limit_layer", default="-1", type=str, required=False,
                        help="The layer for limit training.")
    parser.add_argument("--train_routine",
                        choices=['two_stage', 'all', 'alternate',
                                 'raw', 'self_distil', 'limit',
                                 'weight-linear',
                                 'alternate-lte',
                                 ],
                        default='raw', type=str,
                        help="Training routine (a routine can have mutliple stages, each with different strategies.")

    parser.add_argument("--no_comet", action='store_true',
                        help="Don't upload to comet:highway")
    parser.add_argument("--testset", action='store_true',
                        help="Output results on the test set")
    parser.add_argument('--logging_steps', type=int, default=50,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=50,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument("--log_id", type=str, required=True,
                        help="x for logs/x.log and logs/x.slurm_out (if not interactive")
    parser.add_argument('--server_ip', type=str, default='', help="For distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="For distant debugging.")
    args = parser.parse_args()

    return args
    

args = get_args()

logging.basicConfig(filename="logs/{}.log".format(args.log_id),
                    filemode='w',
                    level=0)
logger = logging.getLogger(__name__)
logger.info("SLURM_JOB_ID: {}".format(os.environ["SLURM_JOB_ID"] if "SLURM_JOB_ID" in os.environ else -1))
logger.info("SLURM_INFO: {}".format([x for x in os.environ.items() if "SLURM" in x[0]]))

# experiment = cm.Experiment(project_name='useless-debug' if args.no_comet else 'highway',
#                            log_code=False,
#                            auto_output_logging=False,
#                            parse_args=False,
#                            auto_metric_logging=False,
#                            display_summary=0)
# experiment.set_name(args.log_id + '--' + str(datetime.date.today()))
# experiment.log_parameters({
#     "log_id": args.log_id,
#     "slurm_id": os.environ["SLURM_JOB_ID"] if "SLURM_JOB_ID" in os.environ else -1
# })


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def get_wanted_result(result):
    if "spearmanr" in result:
        print_result = result["spearmanr"]
    elif "f1" in result:
        print_result = result["f1"]
    elif "mcc" in result:
        print_result = result["mcc"]
    elif "acc" in result:
        print_result = result["acc"]
    else:
        print(result)
        exit(1)
    return print_result


def cal_num_parameters(model):
    counter = {
        "embedding": 0,
        "layernorm": 0,
        "trm": 0,
        "highway": 0,
        "final": 0,
        "all": 0
    }
    for n, p in model.named_parameters():
        size = p.numel()
        if "highway" in n:
            counter["highway"] += size
        elif "layer" in n:
            counter["trm"] += size
        elif "LayerNorm" in n:
            counter["layernorm"] += size
        elif "embedding" in n:
            counter["embedding"] += size
        else:
            print(n)
            counter["final"] += size
        counter["all"] += size
    return counter


def train(args, train_dataset, model, tokenizer, train_strategy='raw'):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        pass
        # tb_writer = SummaryWriter()

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # cal_num_parameters(model)

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    if train_strategy == 'raw':
        # the original bert model
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if
                        ("highway" not in n) and (not any(nd in n for nd in no_decay))],
             'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if
                        ("highway" not in n) and (any(nd in n for nd in no_decay))],
             'weight_decay': 0.0}
        ]
    elif train_strategy == "only_highway":
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if
                        ("highway" in n) and (not any(nd in n for nd in no_decay))],
             'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if
                        ("highway" in n) and (any(nd in n for nd in no_decay))],
             'weight_decay': 0.0}
        ]
    elif train_strategy in ['all', 'self_distil', 'alternate', 'limit',
                            'weight-linear']\
            or train_strategy.endswith('-lte'):
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if
                        not any(nd in n for nd in no_decay)],
             'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if
                        any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}
        ]
    else:
        raise NotImplementedError("Wrong training strategy!")

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                num_training_steps=t_total)

    if args.fp16:
        # haven't fixed for multiple optimizers yet!
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps * (
                    torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss = 0.0
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
    set_seed(args)  # Added here for reproductibility (even between python 2 and 3)

    if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir)
    fout = open(args.output_dir + "/layer_example_counter", 'w')

    print_loss_switch = False  # only True for debugging
    tqdm_disable = print_loss_switch or (args.local_rank not in [-1, 0])

    for epoch_num in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=tqdm_disable)
        layer_example_counter = {i: 0 for i in range(model.num_layers + 1)}
        cumu_loss = 0.0
        epoch_loss = 0.0
        for step, batch in enumerate(epoch_iterator):
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      'labels': batch[3]}
            if args.model_type != 'distilbert':
                inputs['token_type_ids'] = batch[2] if args.model_type == 'bert' else None
                # XLM, DistilBERT and RoBERTa don't use segment_ids
            if train_strategy=='limit':
                inputs['train_strategy'] = train_strategy + args.limit_layer
            else:
                inputs['train_strategy'] = train_strategy
            inputs['layer_example_counter'] = layer_example_counter
            inputs['step_num'] = step
            outputs = model(**inputs)
            loss = outputs[0]  # model outputs are always tuple in transformers (see doc)

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward(retain_graph=True)
            else:
                loss.backward(retain_graph=True)
            tr_loss += loss.item()

            if print_loss_switch and step%10==0:
                print(cumu_loss/10)
                cumu_loss = 0
            cumu_loss += loss.item()

            epoch_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                global_step += 1
                model.zero_grad()

                # save model mid-training - not used for now

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break

        print('Epoch loss: ', epoch_loss)

        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break
    fout.close()
    if args.local_rank in [-1, 0]:
        pass
        # tb_writer.close()

    return global_step, tr_loss / global_step


def evaluate(args, model, tokenizer, prefix="", output_layer=-1, eval_highway=False):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_task_names = ("mnli", "mnli-mm") if args.task_name == "mnli" else (args.task_name,)
    eval_outputs_dirs = (args.output_dir, args.output_dir + '-MM') if args.task_name == "mnli" else (args.output_dir,)

    results = {}
    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        eval_dataset = load_and_cache_examples(args, eval_task, tokenizer,
                                               evaluate=True, testset=args.testset)

        if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(eval_output_dir)

        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

        # multi-gpu eval
        if args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        if args.train_routine.endswith('lte') and args.lte_th == '0.0':
            # create an empty file
            open(
                args.plot_data_dir + args.output_dir + '/uncertainty.txt',
                'w'
            ).close()

        # Eval!
        logger.info("***** Running evaluation {} *****".format(prefix))
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", args.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None
        exit_layer_counter = {(i + 1): 0 for i in range(model.num_layers)}
        entropy_collection = []
        maxlogit_collection = []
        st = time.time()
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            model.eval()
            batch = tuple(t.to(args.device) for t in batch)

            with torch.no_grad():
                inputs = {'input_ids': batch[0],
                          'attention_mask': batch[1],
                          'labels': batch[3]}
                if args.model_type != 'distilbert':
                    inputs['token_type_ids'] = batch[2] if args.model_type == 'bert' else None
                    # XLM, DistilBERT and RoBERTa don't use segment_ids
                if output_layer >= 0:
                    inputs['output_layer'] = output_layer
                outputs = model(**inputs)
                if eval_highway:
                    entropy_collection.append(
                        [x.cpu().item() for x in outputs[3][1][:-1]] + [outputs[3][0].cpu().item()]
                    )
                    maxlogit_collection.append(
                        [torch.max(torch.softmax(x[0], dim=1)).cpu().item() for x in outputs[2]['highway'][:-1]] +\
                        [torch.max(torch.softmax(outputs[1], dim=1)).cpu().item()]
                    )
                    exit_layer_counter[outputs[-1]] += 1
                tmp_eval_loss, logits = outputs[:2]

                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs['labels'].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)

        eval_time = time.time() - st
        print("Eval time:", eval_time)

        if eval_highway and args.early_exit_entropy==-1 and not args.do_train:
            # also record correctness per layer
            save_path = args.plot_data_dir + \
                         args.model_name_or_path[2:]
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            np.save(save_path + "/entropy_distri.npy", np.array(entropy_collection))
            np.save(save_path + "/maxlogit_distri.npy", np.array(maxlogit_collection))
            np.save(save_path + "/correctness_layer{}.npy".format(output_layer),
                    np.array(np.argmax(preds, axis=1) == out_label_ids))
            np.save(save_path + "/prediction_layer{}.npy".format(output_layer),
                    np.array(np.argmax(preds, axis=1)))

        eval_loss = eval_loss / nb_eval_steps
        if args.output_mode == "classification":
            preds = np.argmax(preds, axis=1)
        elif args.output_mode == "regression":
            preds = np.squeeze(preds)
        result = compute_metrics(eval_task, preds, out_label_ids)
        results.update(result)

        output_eval_file = os.path.join(eval_output_dir, prefix, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results {} *****".format(prefix))
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

        if eval_highway:
            print("Exit layer counter", exit_layer_counter)
            actual_cost = sum([l * c for l, c in exit_layer_counter.items()])
            full_cost = len(eval_dataloader) * model.num_layers
            print("Expected saving", actual_cost / full_cost)

            if model.core.encoder.use_lte:
                if not args.do_train:
                    lte_save_fname = args.plot_data_dir + \
                                       args.output_dir + \
                                       "/lte.npy"
                    if args.testset:
                        lte_save_fname = args.plot_data_dir + \
                                         args.output_dir + \
                                         "/lte-test.npy"
                    if not os.path.exists(os.path.dirname(lte_save_fname)):
                        os.makedirs(os.path.dirname(lte_save_fname))
                    if not os.path.exists(lte_save_fname):
                        prev_saver = []
                    else:
                        prev_saver = np.load(lte_save_fname, allow_pickle=True).tolist()

                    print_result = get_wanted_result(result)
                    prev_saver.append([
                        exit_layer_counter,
                        eval_time,
                        actual_cost / full_cost,
                        print_result,
                        {'lte_th': model.core.encoder.lte_th}
                    ])
                    np.save(lte_save_fname, np.array(prev_saver))
                    # experiment.log_metrics({
                    #     "eval_time": eval_time,
                    #     "ERS": actual_cost / full_cost,
                    #     "avg-layer": actual_cost / full_cost * model.num_layers,
                    #     "result": print_result,
                    # })
                    # experiment.log_other(
                    #     "exit_layer_counter",
                    #     str(exit_layer_counter),
                    # )

            elif args.early_exit_entropy >= -0.5:
                save_fname = args.plot_data_dir + \
                             args.model_name_or_path[2:] + \
                             ("/testset/" if args.testset else "") + \
                             "/entropy_{}.npy".format(args.early_exit_entropy)
                if not os.path.exists(os.path.dirname(save_fname)):
                    os.makedirs(os.path.dirname(save_fname))
                print_result = get_wanted_result(result)
                np.save(save_fname,
                        np.array([exit_layer_counter,
                                  eval_time,
                                  actual_cost / full_cost,
                                  print_result]))
                # experiment.log_metrics({
                #     "eval_time": eval_time,
                #     "ERS": actual_cost / full_cost,
                #     "avg-layer": actual_cost / full_cost * model.num_layers,
                #     "result": print_result,
                # })
                # experiment.log_other(
                #     "exit_layer_counter",
                #     str(exit_layer_counter),
                # )

            if args.testset:
                label_list = processors[eval_task]().get_labels()
                eval_task_name = eval_task.upper()
                if eval_task_name == 'MNLI':
                    eval_task_name = 'MNLI-m'
                elif eval_task_name == 'MNLI-MM':
                    eval_task_name = 'MNLI-mm'
                if 'MNLI' in eval_task_name:
                    # label_list before swapping: contradiction, entailment, neutral
                    # I'm still very confused
                    label_list = ['contradiction', 'entailment', 'neutral']

                marker = '3' if eval_task_name=='STS-B' else args.early_exit_entropy
                submit_fname = args.plot_data_dir + \
                    args.model_name_or_path[2:] + \
                    "/testset/{}-{}.tsv".format(marker, eval_task_name)
                if not os.path.exists(os.path.dirname(submit_fname)):
                    os.makedirs(os.path.dirname(submit_fname))
                with open(submit_fname, 'w') as fout:
                    print("index\tprediction", file=fout)
                    for i, p in enumerate(preds):
                        if eval_task_name != 'STS-B':
                            print('{}\t{}'.format(i, label_list[p]), file=fout)
                        else:
                            print('{}\t{:.3f}'.format(i, p), file=fout)
    return results


def load_and_cache_examples(args, task, tokenizer, evaluate=False, testset=False):
    if args.local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    processor = processors[task]()
    output_mode = output_modes[task]
    # Load data features from cache or dataset file
    split = 'train'
    if evaluate:
        split = 'dev'
    if testset:
        split = 'test'
    cached_features_file = os.path.join(args.data_dir, 'cached_{}_{}_{}_{}'.format(
        split,
        list(filter(None, args.model_name_or_path.split('/'))).pop(),
        str(args.max_seq_length),
        str(task)))
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        label_list = processor.get_labels()
        if task in ['mnli', 'mnli-mm'] and args.model_type in ['roberta']:
            # HACK(label indices are swapped in RoBERTa pretrained model)
            label_list[1], label_list[2] = label_list[2], label_list[1]
            # after swap: contradiction, neutral, entailment
        if not evaluate:
            examples = processor.get_train_examples(args.data_dir)
        elif not testset:
            examples = processor.get_dev_examples(args.data_dir)
        else:
            examples = processor.get_test_examples(args.data_dir)
        features = convert_examples_to_features(examples,
                                                tokenizer,
                                                label_list=label_list,
                                                max_length=args.max_seq_length,
                                                output_mode=output_mode,
                                                pad_on_left=bool(args.model_type in ['xlnet']),
                                                # pad on the left for xlnet
                                                pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                                                pad_token_segment_id=4 if args.model_type in ['xlnet'] else 0,
                                                )
        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)

    if args.local_rank == 0 and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    if output_mode == "classification":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
    elif output_mode == "regression":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.float)

    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)
    return dataset


def main(args):

    # experiment.log_parameters(vars(args))
    if 'saved_models' in args.model_name_or_path:  # evaluation
        model_and_size = args.model_name_or_path[
                         args.model_name_or_path.find('saved_models') + 13:]
        model_and_size = model_and_size[
                         :model_and_size.find('/')
                         ]
    else:  # training
        flag = args.model_name_or_path.find('-uncased')  # bert
        if flag == -1:
            model_and_size = args.model_name_or_path
        else:
            model_and_size = args.model_name_or_path[:flag]

        flag = args.model_name_or_path.find('-v2')  # albert
        if flag == -1:
            model_and_size = model_and_size
        else:
            model_and_size = model_and_size[:flag]
    # experiment.log_parameter(
    #     "model_and_size",
    #     model_and_size
    # )

    if args.train_routine == 'limit':
        finished_layers = os.listdir(args.plot_data_dir + args.output_dir)
        for fname in finished_layers:
            layer = fname[len('layer-'):fname.index('.npy')]
            try:
                if layer == args.limit_layer:  # both are type'str'
                    # already done
                    exit(0)
            except ValueError:
                pass

    if os.path.exists(args.output_dir) and os.listdir(
            args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir))

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    # Set seed
    set_seed(args)

    # Prepare GLUE task
    args.task_name = args.task_name.lower()
    if args.task_name not in processors:
        raise ValueError("Task not found: %s" % (args.task_name))
    processor = processors[args.task_name]()
    args.output_mode = output_modes[args.task_name]
    label_list = processor.get_labels()
    num_labels = len(label_list)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                          num_labels=num_labels,
                                          finetuning_task=args.task_name,
                                          cache_dir=args.cache_dir if args.cache_dir else None)

    config.divide = args.train_routine

    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
                                                do_lower_case=args.do_lower_case,
                                                cache_dir=args.cache_dir if args.cache_dir else None)
    model = model_class.from_pretrained(args.model_name_or_path,
                                        from_tf=bool('.ckpt' in args.model_name_or_path),
                                        config=config,
                                        cache_dir=args.cache_dir if args.cache_dir else None)

    model.core.encoder.set_early_exit_entropy(args.early_exit_entropy)
    model.core.init_highway_pooler()
    if args.train_routine.endswith("-lte"):
        model.core.encoder.enable_lte(args)


    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        train_dataset = load_and_cache_examples(args, args.task_name, tokenizer, evaluate=False)

        if args.train_routine == "two_stage":
            # first stage
            global_step, tr_loss = train(args, train_dataset, model, tokenizer,
                                         train_strategy='raw')
            logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)
            result = evaluate(args, model, tokenizer, prefix="")
            print_result = get_wanted_result(result)
            print("result: {}".format(print_result))
            # experiment.log_metric("Result after first stage training", print_result)

            # second stage
            global_step, tr_loss = train(args, train_dataset, model, tokenizer,
                                         train_strategy="only_highway")

        elif args.train_routine in ['raw', 'all', 'alternate',
                                    'self_distil', 'limit',
                                    'weight-linear',] \
            or args.train_routine.endswith('-lte'):

            global_step, tr_loss = train(args, train_dataset, model, tokenizer,
                                         train_strategy=args.train_routine)
            logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

        else:
            raise NotImplementedError("Wrong training routine!")

    # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Create output directory if needed
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)

        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = model.module if hasattr(model,
                                                'module') else model  # Take care of distributed/parallel training
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))

        # Load a trained model and vocabulary that you have fine-tuned
        model = model_class.from_pretrained(args.output_dir)
        tokenizer = tokenizer_class.from_pretrained(args.output_dir)
        model.to(args.device)

    # Evaluation
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + '/**/' + WEIGHTS_NAME, recursive=True)))
            logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split('-')[-1] if len(checkpoints) > 1 else ""
            prefix = checkpoint.split('/')[-1] if checkpoint.find('checkpoint') != -1 else ""

            model = model_class.from_pretrained(checkpoint)
            model.core.encoder.set_early_exit_entropy(args.early_exit_entropy)
            model.to(args.device)
            if args.train_routine.endswith('-lte'):
                model.core.encoder.enable_lte(args)
                args.eval_highway = True  # triggers ERS measurement

            result = evaluate(args, model, tokenizer, prefix=prefix,
                              eval_highway=args.eval_highway,
                              output_layer=int(args.limit_layer))
            print_result = get_wanted_result(result)
            print("result: {}".format(print_result))
            # experiment.log_metric("final result", print_result)
            if args.train_routine=='limit':
                save_fname = args.plot_data_dir + \
                             args.output_dir + f"/layer-{args.limit_layer}.npy"
                if not os.path.exists(os.path.dirname(save_fname)):
                    os.makedirs(os.path.dirname(save_fname))
                np.save(save_fname, np.array([print_result]))

            if args.eval_each_highway and args.lte_th=='-1':
                last_layer_results = print_result
                each_layer_results = []
                for i in range(model.num_layers-1):
                    logger.info("\n")
                    _result = evaluate(args, model, tokenizer, prefix=prefix,
                                       output_layer=i, eval_highway=args.eval_highway)
                    each_layer_results.append(get_wanted_result(_result))
                each_layer_results.append(last_layer_results)
                # experiment.log_other(
                #     "Each layer result",
                #     ' '.join(['{:.0f}'.format(100*x) for x in each_layer_results]))
                save_fname = args.plot_data_dir + args.model_name_or_path[2:] + "/each_layer.npy"
                if not os.path.exists(os.path.dirname(save_fname)):
                    os.makedirs(os.path.dirname(save_fname))
                np.save(save_fname,
                        np.array(each_layer_results))
            result = dict((k + '_{}'.format(global_step), v) for k, v in result.items())
            results.update(result)

    return results


if __name__ == "__main__":
    main(args)
