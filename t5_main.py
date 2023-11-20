import os
import sys
import math
import json
import tqdm
import torch
import random
import argparse
import my_logging as logging
import shutil
import numpy as np
from collections import Counter
from transformers import T5ForConditionalGeneration, BartForConditionalGeneration
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import T5Tokenizer, BartTokenizer

torch.backends.cudnn.enable =True
torch.backends.cudnn.benchmark = True

def main():
    args = parse_args()
    if args.eval_only:
        do_eval(args)
    elif args.pretrain:
        pretrain(args)
    else:
        do_train(args)

def parse_args():
    """
    Argument settings.
    """
    # arguments for experiments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pretrain",
        action="store_true",
        help="few-shot pretraining."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2023,
        help="Random seed for reproducing experimental results",
    )
    parser.add_argument(
        "--margin", type=float, required=False, default=0.3, help="Pairwise loss margin"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="conll",
        choices=["conll", "movie", "restaurant", "multiwoz"],
        help="The name (id) of dataset",
    )
    parser.add_argument(
        "--training",
        action="store_true",
        help="training steps",
    )
    parser.add_argument(
        "--testing",
        action="store_true",
        help="testing steps",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="./bert-base-uncased/",
        help="The model path or model name of pre-trained model",
    )
    parser.add_argument(
        "--continue_training_path",
        type=str,
        default=None,
        help="The checkpoint path to load for continue training",
    )
    parser.add_argument("--model_save_path", type=str, help="Custom output dir")
    parser.add_argument(
        "--force_del",
        action="store_true",
        help="Delete the existing save_path and do not report an error",
    )
    parser.add_argument("--eval_only", action="store_true", help="Only do evaluation")
    parser.add_argument(
        "--early_stop", action="store_true", help="Stop if performance don't increase"
    )
    parser.add_argument(
        "--train_file_path",
        type=str,
        default="train.json",
        help="The filename of train dataset",
    )
    parser.add_argument(
        "--dev_file_path",
        type=str,
        default="dev.json",
        help="The filename of dev dataset",
    )
    parser.add_argument(
        "--test_file_path",
        type=str,
        default="test.json",
        help="The filename of test dataset",
    )

    # arguments for models
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="The dropout rate of T5 derived representations",
    )

    # arguments for training (or data preprocessing)
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Training mini-batch size"
    )
    parser.add_argument(
        "--eval_batch_size", type=int, default=128, help="Training mini-batch size"
    )
    parser.add_argument(
        "--num_epochs", type=int, default=1, help="Number of training epochs"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=5e-5, help="The learning rate"
    )
    parser.add_argument(
        "--evaluation_steps",
        type=int,
        default=1000,
        help="The steps between every evaluations",
    )
    parser.add_argument(
        "--max_seq_length", type=int, default=256, help="The max sequence length"
    )
    parser.add_argument(
        "--num", type=int, default=2, help="demons number"
    )
    parser.add_argument(
        "--decode_max_seq_length", type=int, default=128, help="The max sequence length"
    )
    parser.add_argument(
        "--gradient_clip",
        type=float,
        default=1.0,
        help="The max norm of gradient to apply gradient clip",
    )
    parser.add_argument(
        "--patience", type=int, default=None, help="Patience for early stop"
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="f1",
        choices=["f1", "loss"],
        help="The metric to select the best model",
    )
    parser.add_argument(
        "--clean_train_path", type=str, default=None, help="clean train path"
    )
    parser.add_argument(
        "--noise_train_path", type=str, default=None, help="noise train path"
    )
    parser.add_argument(
        "--clean_dev_path", type=str, default=None, help="clean dev path"
    )
    parser.add_argument(
        "--noise_dev_path", type=str, default=None, help="noise dev path"
    )
    parser.add_argument("--eval_loss", type=float, default=0.2, help="evaluate min loss")
    parser.add_argument(
        "--auxiliary_without_eval",
        action="store_true",
        help="do auxilary task without evaluate.",
    )
    parser.add_argument(
        "--logging_file_path", type=str, help="The file-path to logging files"
    )
    parser.add_argument(
        "--add_demonstration",
        action="store_true",
        help="Use some demonstrations to enhance the model.",
    )
    parser.add_argument(
        "--demons_train_path", type=str, default=None, help="Demonstration file path."
    )
    parser.add_argument(
        "--demons_out_path", type=str, default=None, help="Demonstration out path."
    )
    parser.add_argument(
        "--demons_valid_path",
        type=str,
        default=None,
        help="Demonstration valid file path.",
    )
    parser.add_argument(
        "--demons_val_out_path",
        type=str,
        default=None,
        help="Demonstration valid out path.",
    )

    # 多任务路径参数
    parser.add_argument("--mask_output_path", type=str, help="mask task output file.")
    parser.add_argument("--mask_input_path", type=str, help="mask task input file.")
    parser.add_argument(
        "--classify_output_path", type=str, help="classify task output file."
    )
    parser.add_argument(
        "--classify_input_path", type=str, help="classify task input file."
    )
    parser.add_argument("--clean_path", type=str, help="noise task input file.")
    parser.add_argument("--noise_path", type=str, help="noise task output file.")
    args = parser.parse_args()

    return args

class T5NERDataLoader(object):
    def __init__(
        self,
        args,
        file_path,
        split,
        tokenizer, 
        max_seq_length,
        decode_max_seq_length,
        dataset,
        add_demonstration,
        demons_train_path,
        demons_out_path,
        shuffle: bool = False,
    ):
        logging.info(f"Reading {split} split from {file_path}")
        lines = []
        self.args = args
        tmp_sample, tmp_label = [], []
        label_freq = Counter()
        self.dataset = dataset
        self.max_length = max_seq_length
        self.decode_max_seq_length = decode_max_seq_length
        f = open(file_path, "r")
        if add_demonstration:
            f1 = open(demons_train_path, "r")
            f2 = open(demons_out_path, "r")

        for ln in f:
            ln = ln.strip()
            if len(ln) == 0:
                if len(tmp_sample) == 0:
                    continue
                assert len(tmp_sample) == len(tmp_label)
                entitiesnlabels = get_sentence_ner(
                    tmp_sample, tmp_label, span_only=False
                )
                target_text = []
                for entity_label in entitiesnlabels:
                    entity, label = entity_label.split("||")
                    label = label_to_language(label)
                    target_text.append(entity + label)

                if add_demonstration:
                    ln1 = f1.readline()
                    ln2 = f2.readline()
                    demons_train = []
                    demons_target = []
                    tr_text = ""
                    if demons_out_path is None or demons_train_path is None:
                        raise NameError("Please provide the demons file path.")

                    ln1 = ln1.strip().split(" ||| ")
                    ln2 = ln2.strip().split(" ||| ")
                    assert len(ln1) == len(ln2)
                    for l1, l2 in zip(ln1, ln2):
                        demons_train = l1.strip().split(" ")
                        demons_target = l2.strip().split(" ")
                        en_la = get_sentence_ner(
                            demons_train, demons_target, span_only=False
                        )
                        demons_text = []
                        for ent_lbl in en_la:
                            en, la = ent_lbl.split("||")
                            la = label_to_language(la)
                            demons_text.append(en + la)
                        tr_text += " ".join(demons_train)
                        # tr_text += " In the above sentence : "
                        # tr_text += " , ".join(demons_text)
                        tr_text += ". "
                        demons_train, demons_target = [], []
                    lines.append(
                        (
                            tr_text
                            + " ".join(tmp_sample)
                            + f" In the above sentence : ",
                            " , ".join(target_text),
                        )
                    )
                    tr_text = ""
                else:
                    lines.append(
                        (
                            " ".join(tmp_sample) + " In the above sentence : ",
                            " , ".join(target_text),
                        )
                    )
                tmp_sample, tmp_label = [], []
            else:
                smp, lbl = ln.split()
                tmp_sample.append(smp)
                tmp_label.append(lbl)
                label_freq[lbl] += 1

        logging.info(f"Number of samples: total={len(lines)}")
        for k, v in label_freq.items():
            logging.info(f"{k}={v} ({100 * v / len(lines):.2f}%)")

        if shuffle:
            random.shuffle(lines)
        
        self.tokenizer = tokenizer
        self.data = lines
        self.num_data = len(self.data)

    def shuffle(self):
        random.shuffle(self.data)

    def get_batch(self, iteration, batch_size):
        batch_data = self.data[iteration * batch_size : (iteration + 1) * batch_size]
        
        if "gpt" in self.args.model_name_or_path:
            bert_inputs = self.tokenizer.batch_encode_plus(
                [sample[0] + "; " + sample[1] for sample in batch_data],
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            targets = self.tokenizer.batch_encode_plus(
                [sample[0] + "; " + sample[1] for sample in batch_data],
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
        else:
            bert_inputs = self.tokenizer.batch_encode_plus(
                [sample[0] for sample in batch_data],
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            targets = self.tokenizer.batch_encode_plus(
                [sample[1] for sample in batch_data],
                padding=True,
                truncation=True,
                max_length=self.decode_max_seq_length,
                return_tensors="pt",
            )

        input_ids = bert_inputs["input_ids"]
        input_mask = bert_inputs["attention_mask"]
        label_ids = targets["input_ids"]
        sentences = [sample[0] for sample in batch_data]
        entities = [sample[1] for sample in batch_data]

        return input_ids, input_mask, label_ids, sentences, entities

class T5AuxiliaryDataLoader(object):
    def __init__(
        self,
        args,
        clean_file_path,
        noise_file_path,
        split,
        tokenizer, 
        max_seq_length,
        decode_max_seq_length,
        dataset,
        shuffle=False,
    ):
        self.args = args
        logging.info(
            f"Reading {split} split from {clean_file_path} and {noise_file_path}"
        )
        lines = []
        self.dataset = dataset
        self.max_length = max_seq_length
        if "gpt" in args.model_name_or_path:
            self.decode_max_seq_length = max_seq_length
        else:
            self.decode_max_seq_length = decode_max_seq_length
        with open(clean_file_path, "r") as f_clean, open(
            noise_file_path, "r"
        ) as f_noise:
            for ln_out, ln_in in zip(f_clean, f_noise):
                ln_out = ln_out.strip()
                ln_in = ln_in.strip()
                if len(ln_out) == 0 or len(ln_in) == 0:
                    continue
                ln_in = " ".join(ln_in)
                lines.append((ln_in, ln_out))
        if shuffle:
            random.shuffle(lines)
        self.tokenizer = tokenizer
        self.data = lines
        self.num_data = len(self.data)

    def shuffle(self):
        random.shuffle(self.data)

    def get_batch(self, iteration, batch_size):
        batch_data = self.data[iteration * batch_size : (iteration + 1) * batch_size]

        if "gpt" in self.args.model_name_or_path:
            bert_inputs = self.tokenizer.batch_encode_plus(
                [sample[0] + "; " + sample[1] for sample in batch_data],
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            targets = self.tokenizer.batch_encode_plus(
                [sample[0] + "; " + sample[1] for sample in batch_data],
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
        else:
            bert_inputs = self.tokenizer.batch_encode_plus(
                [sample[0] for sample in batch_data],
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            targets = self.tokenizer.batch_encode_plus(
                [sample[1] for sample in batch_data],
                padding=True,
                truncation=True,
                max_length=self.decode_max_seq_length,
                return_tensors="pt",
            )

        input_ids = bert_inputs["input_ids"]
        input_mask = bert_inputs["attention_mask"]
        label_ids = targets["input_ids"]
        
        noises = [sample[0] for sample in batch_data]
        cleans = [sample[1] for sample in batch_data]

        return input_ids, input_mask, label_ids, noises, cleans

def load_model_and_tokenizer(args):
    # 加载模型
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if "t5" in args.model_name_or_path:
        model = T5ForConditionalGeneration.from_pretrained(args.model_name_or_path).to(device)
        tokenizer = T5Tokenizer.from_pretrained(args.model_name_or_path)
    if "bart" in args.model_name_or_path:
        model = BartForConditionalGeneration.from_pretrained(args.model_name_or_path).to(device)
        tokenizer = BartTokenizer.from_pretrained(args.model_name_or_path)
    if "gpt" in args.model_name_or_path:
        model = GPT2LMHeadModel.from_pretrained(args.model_name_or_path).to(device)
        tokenizer = GPT2Tokenizer.from_pretrained(args.model_name_or_path)
        tokenizer.pad_token = tokenizer.eos_token
        
    if args.continue_training_path is not None:
        model.load_state_dict(
            torch.load(os.path.join(args.continue_training_path, "pytorch_model.bin")),
            strict=False,
        )
    
    return model, tokenizer

def pretrain(args):
    check_args(args)
    prepare_for_training(args)
    model, tokenizer = load_model_and_tokenizer(args)
    
    mask_loader = T5AuxiliaryDataLoader(
        args,
        args.mask_output_path,
        args.mask_input_path,
        "mask_task",
        tokenizer,
        args.max_seq_length,
        args.decode_max_seq_length,
        args.dataset,
    )

    noise_loader = T5AuxiliaryDataLoader(
        args,
        args.clean_path,
        args.noise_path,
        "noise_to_clean_task",
        tokenizer,
        args.max_seq_length,
        args.decode_max_seq_length,
        args.dataset,
    )

    classify_loader = T5AuxiliaryDataLoader(
        args,
        args.classify_output_path,
        args.classify_input_path,
        "classify_task",
        tokenizer,
        args.max_seq_length,
        args.decode_max_seq_length,
        args.dataset,
    )

    trainable_parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_parameters, lr=args.learning_rate)

    logging.info(f"Start training, number of epochs: {args.num_epochs}")
    global_step = 0
    # 训练
    for epoch_idx in range(args.num_epochs):
        logging.info(f"Training start for epoch {epoch_idx}")
        sum_train_loss = 0.0
        sum_train_acc = 0.0
        num_batches = math.ceil(noise_loader.num_data / args.batch_size)
        logging.info(f"Total number of batches: {num_batches}")
        with tqdm.trange(num_batches) as trange_obj:
            for batch_idx in trange_obj:
                trange_obj.set_postfix(
                    loss=f"{sum_train_loss / batch_idx if batch_idx > 0 else 0.0:10.2f}",
                    acc=f"{sum_train_acc / batch_idx if batch_idx > 0 else 0.0:10.2f}",
                )
                model.train()
            
                batch_mask = mask_loader.get_batch(batch_idx, args.batch_size)
                batch_mask = [
                    item.cuda() if isinstance(item, torch.Tensor) else item
                    for item in batch_mask
                ]
                input_ids, input_mask, label_ids, _, _ = batch_mask

                output_mask = model(
                    input_ids=input_ids,
                    attention_mask=input_mask,
                    labels=label_ids,
                )
                loss_mask = output_mask.loss
                
                # classify任务的batch
                batch_classify = classify_loader.get_batch(batch_idx, args.batch_size)
                batch_classify = [
                    item.cuda() if isinstance(item, torch.Tensor) else item
                    for item in batch_classify
                ]
                input_ids, input_mask, label_ids, _, _ = batch_classify

                output_classify = model(
                    input_ids=input_ids,
                    attention_mask=input_mask,
                    labels=label_ids,
                )
                loss_classify = output_classify.loss

                # noise任务的batch
                batch_noise = noise_loader.get_batch(batch_idx, args.batch_size)
                batch_noise = [
                    item.cuda() if isinstance(item, torch.Tensor) else item
                    for item in batch_noise
                ]
                input_ids, input_mask, label_ids, _, _ = batch_noise

                output_noise = model(
                    input_ids=input_ids,
                    attention_mask=input_mask,
                    labels=label_ids,
                )
                loss_noise = output_noise.loss

                loss = loss_mask + loss_classify + loss_noise
                
                sum_train_loss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.gradient_clip
                )
                optimizer.step()
                global_step += 1

        model_save_file = os.path.join(args.model_save_path, "pytorch_model.bin")
        logging.info(
            f"One epoch finished, epoch={epoch_idx}, save model to {model_save_file}"
        )
        torch.save(model.state_dict(), model_save_file)

def do_train(args):
    check_args(args)
    prepare_for_training(args)
    
    model, tokenizer = load_model_and_tokenizer(args)

    train_loader = T5NERDataLoader(
        args,
        args.train_file_path,
        "train",
        tokenizer,
        args.max_seq_length,
        args.decode_max_seq_length,
        args.dataset,
        add_demonstration=args.add_demonstration,
        demons_train_path=args.demons_train_path,
        demons_out_path=args.demons_out_path,
    )
    dev_loader = T5NERDataLoader(
        args,
        args.dev_file_path,
        "dev",
        tokenizer,
        args.max_seq_length,
        args.decode_max_seq_length,
        args.dataset,
        add_demonstration=args.add_demonstration,
        demons_train_path=args.demons_valid_path,
        demons_out_path=args.demons_val_out_path,
    )
        
    # 一些参数、优化器等
    trainable_parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_parameters, lr=args.learning_rate)

    logging.info(f"Start training, number of epochs: {args.num_epochs}")
    global_step = 0
    best_dev_performance = 0.0
    early_stop_step = 0

    # 训练
    for epoch_idx in range(args.num_epochs):
        logging.info(f"Training start for epoch {epoch_idx}")
        sum_train_loss = 0.0
        sum_train_acc = 0.0
        num_batches = math.ceil(train_loader.num_data / args.batch_size)
        logging.info(f"Total number of batches: {num_batches}")

        with tqdm.trange(num_batches) as trange_obj:
            for batch_idx in trange_obj:
                trange_obj.set_postfix(
                    loss=f"{sum_train_loss / batch_idx if batch_idx > 0 else 0.0:10.2f}",
                    acc=f"{sum_train_acc / batch_idx if batch_idx > 0 else 0.0:10.2f}",
                )

                model.train()
                tokenizer.padding_side = "right"
                batch = train_loader.get_batch(batch_idx, args.batch_size)
                batch = [
                    item.cuda() if isinstance(item, torch.Tensor) else item
                    for item in batch
                ]
                input_ids, input_mask, label_ids, _, _ = batch

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=input_mask,
                    labels=label_ids,
                )

                loss = outputs.loss

                sum_train_loss += loss.item()
                model.zero_grad()
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.gradient_clip
                )
                optimizer.step()
                global_step += 1

                if (
                    global_step + 1
                ) % args.evaluation_steps == 0:
                    logging.info(
                        f"Evaluating on dev dataset at step {batch_idx} in epoch {epoch_idx} (global step: {global_step})..."
                    )
                    eval_metrics = evaluate(
                        args,
                        tokenizer,
                        dev_loader,
                        model,
                        args.eval_batch_size,
                        args.max_seq_length,
                        args.decode_max_seq_length,
                        log_file=os.path.join(
                            args.model_save_path,
                            f"evaluate_dev_step{global_step}.json",
                        ),
                    )
                    current_dev_performance = eval_metrics[args.metric]
                    for k, v in eval_metrics.items():
                        logging.info(f"Dev Results: {k} = {v:.4f}")
                            
                    if current_dev_performance > best_dev_performance:
                        model_save_file = os.path.join(
                            args.model_save_path, "pytorch_model.bin"
                        )
                        logging.info(
                            f"Best performance achieved, epoch={epoch_idx}, save model to {model_save_file}"
                        )
                        torch.save(model.state_dict(), model_save_file)
                        json.dump(
                            eval_metrics,
                            open(
                                os.path.join(
                                    args.model_save_path,
                                    f"evaluate_dev_best.json",
                                ),
                                "w",
                            ),
                            indent=4,
                        )
                        best_dev_performance = current_dev_performance
                        early_stop_step = 0
                    else:
                        early_stop_step += 1

def construct_test_file_dict(test_file_path):
    ret = dict()
    ret["clean"] = test_file_path + "/clean_test/train/test.txt"
    ret["verbose"] = test_file_path + "/verbose/train/test.txt"
    ret["speech"] = test_file_path + "/speech/train/test.txt"
    ret["simplify"] = test_file_path + "/simplify/train/test.txt"
    ret["paraphrase"] = test_file_path + "/paraphrase/train/test.txt"
    ret["typos"] = test_file_path + "/typos/train/test.txt"
    
    return ret

def construct_test_demons_dict(test_file_path, num):
    ret = dict()
    ret["clean"] = (
        test_file_path + "/clean_test/train/demons.in",
        test_file_path + "/clean_test/train/demons.out",
    )
    ret["verbose"] = (
        test_file_path + "/verbose/train/demons.in",
        test_file_path + "/verbose/train/demons.out",
    )
    ret["speech"] = (
        test_file_path + "/speech/train/demons.in",
        test_file_path + "/speech/train/demons.out",
    )
    ret["simplify"] = (
        test_file_path + "/simplify/train/demons.in",
        test_file_path + "/simplify/train/demons.out",
    )
    ret["paraphrase"] = (
        test_file_path + "/paraphrase/train/demons.in",
        test_file_path + "/paraphrase/train/demons.out",
    )
    ret["typos"] = (
        test_file_path + "/typos/train/demons.in",
        test_file_path + "/typos/train/demons.out",
    )
    return ret

def do_eval(args):
    check_args(args)
    prepare_for_eval(args)
    
    test_file_dict = construct_test_file_dict(args.test_file_path)
    test_demons_dict = construct_test_demons_dict(args.test_file_path, args.num)
    model, tokenizer = load_model_and_tokenizer(args)

    # evaluate on clean test dataset
    logging.info("Evaluation on clean test dataset...")
    test_loader = T5NERDataLoader(
        args,
        test_file_dict["clean"],
        "clean",
        tokenizer,
        args.max_seq_length,
        args.decode_max_seq_length,
        args.dataset,
        add_demonstration=args.add_demonstration,
        demons_train_path=test_demons_dict["clean"][0],
        demons_out_path=test_demons_dict["clean"][1],
    )
    eval_metrics = evaluate(
        args,
        tokenizer,
        test_loader,
        model,
        args.eval_batch_size,
        args.max_seq_length,
        args.decode_max_seq_length,
        log_file=os.path.join(args.logging_file_path, "evaluate_test_clean.json"),
    )
    for k, v in eval_metrics.items():
        logging.info(f"Clean test Results: {k} = {v:.4f}")

    # evaluate on verbose dataset
    test_loader_verbose = T5NERDataLoader(
        args,
        test_file_dict["verbose"],
        "verbose",
        tokenizer,
        args.max_seq_length,
        args.decode_max_seq_length,
        args.dataset,
        add_demonstration=args.add_demonstration,
        demons_train_path=test_demons_dict["verbose"][0],
        demons_out_path=test_demons_dict["verbose"][1],
    )
    logging.info("Evaluation on verbose test dataset...")
    eval_metrics_verbose = evaluate(
        args,
        tokenizer,
        test_loader_verbose,
        model,
        args.eval_batch_size,
        args.max_seq_length,
        args.decode_max_seq_length,
        log_file=os.path.join(args.logging_file_path, "evaluate_test_verbose.json"),
    )
    for k, v in eval_metrics_verbose.items():
        logging.info(f"Verbose test Results: {k} = {v:.4f}")

    # evaluate on paraphrase dataset
    test_loader_paraphrase = T5NERDataLoader(
        args,
        test_file_dict["paraphrase"],
        "paraphrase",
        tokenizer,
        args.max_seq_length,
        args.decode_max_seq_length,
        args.dataset,
        add_demonstration=args.add_demonstration,
        demons_train_path=test_demons_dict["paraphrase"][0],
        demons_out_path=test_demons_dict["paraphrase"][1],
    )
    logging.info("Evaluation on paraphrase test dataset...")
    eval_metrics_para = evaluate(
        args,
        tokenizer,
        test_loader_paraphrase,
        model,
        args.eval_batch_size,
        args.max_seq_length,
        args.decode_max_seq_length,
        log_file=os.path.join(args.logging_file_path, "evaluate_test_paraphrase.json"),
    )
    for k, v in eval_metrics_para.items():
        logging.info(f"Paraphrase test Results: {k} = {v:.4f}")

    # evaluate on typos dataset
    test_loader_typos = T5NERDataLoader(
        args,
        test_file_dict["typos"],
        "typos",
        tokenizer,
        args.max_seq_length,
        args.decode_max_seq_length,
        args.dataset,
        add_demonstration=args.add_demonstration,
        demons_train_path=test_demons_dict["typos"][0],
        demons_out_path=test_demons_dict["typos"][1],
    )
    logging.info("Evaluation on typos test dataset...")
    eval_metrics_typos = evaluate(
        args,
        tokenizer,
        test_loader_typos,
        model,
        args.eval_batch_size,
        args.max_seq_length,
        args.decode_max_seq_length,
        log_file=os.path.join(args.logging_file_path, "evaluate_test_typos.json"),
    )
    for k, v in eval_metrics_typos.items():
        logging.info(f"Typos test Results: {k} = {v:.4f}")

    # evaluate on simplify dataset
    test_loader_simplify = T5NERDataLoader(
        args,
        test_file_dict["simplify"],
        "simplify",
        tokenizer,
        args.max_seq_length,
        args.decode_max_seq_length,
        args.dataset,
        add_demonstration=args.add_demonstration,
        demons_train_path=test_demons_dict["simplify"][0],
        demons_out_path=test_demons_dict["simplify"][1],
    )
    logging.info("Evaluation on simplify test dataset...")
    eval_metrics_simplify = evaluate(
        args,
        tokenizer,
        test_loader_simplify,
        model,
        args.eval_batch_size,
        args.max_seq_length,
        args.decode_max_seq_length,
        log_file=os.path.join(args.logging_file_path, "evaluate_test_simplify.json"),
    )
    for k, v in eval_metrics_simplify.items():
        logging.info(f"Simplify test Results: {k} = {v:.4f}")

    # evaluate on speech dataset
    test_loader_speech = T5NERDataLoader(
        args,
        test_file_dict["speech"],
        "speech",
        tokenizer,
        args.max_seq_length,
        args.decode_max_seq_length,
        args.dataset,
        add_demonstration=args.add_demonstration,
        demons_train_path=test_demons_dict["speech"][0],
        demons_out_path=test_demons_dict["speech"][1],
    )
    logging.info("Evaluation on speech test dataset...")
    eval_metrics_speech = evaluate(
        args,
        tokenizer,
        test_loader_speech,
        model,
        args.eval_batch_size,
        args.max_seq_length,
        args.decode_max_seq_length,
        log_file=os.path.join(args.logging_file_path, "evaluate_test_speech.json"),
    )
    for k, v in eval_metrics_speech.items():
        logging.info(f"Speech test Results: {k} = {v:.4f}")

    return (
        eval_metrics,
        eval_metrics_para,
        eval_metrics_speech,
        eval_metrics_simplify,
        eval_metrics_typos,
        eval_metrics_verbose,
    )

def check_args(args):
    assert args.metric in (
        "f1",
        "exact_match",
        "loss",
    ), "KBQA task only support hits and f1 as metric"


def set_seed(seed: int):
    """
    Added script to set random seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True


def prepare_for_training(args):
    """
    - Initialize the random seed
    - Create the output directory
    - Record the training command and arguments
    """
    set_seed(args.seed)

    if os.path.exists(args.model_save_path):
        if args.force_del:
            shutil.rmtree(args.model_save_path)
            os.mkdir(args.model_save_path)
    else:
        os.mkdir(args.model_save_path)

    command_save_filename = os.path.join(args.model_save_path, "command.txt")
    with open(command_save_filename, "w") as f:
        CUDA_VISIBLE_DEVICES = os.environ.get("CUDA_VISIBLE_DEVICES")
        if CUDA_VISIBLE_DEVICES is not None:
            f.write(
                f"CUDA_VISIBLE_DEVICES={CUDA_VISIBLE_DEVICES} python3 {' '.join(sys.argv)}"
            )
        else:
            f.write(f"python3 {' '.join(sys.argv)}")

    args_save_filename = os.path.join(args.model_save_path, "args.json")
    with open(args_save_filename, "w") as f:
        json.dump(args.__dict__, f, indent=4, ensure_ascii=False)

    logging_file = os.path.join(args.model_save_path, "stdout.log")
    logging_stream = open(logging_file, "a")
    logging.basicConfig(logging_stream)

    logging.info(
        f"Training arguments:\n{json.dumps(args.__dict__, indent=4, ensure_ascii=False)}"
    )
    logging.info(f"Run experiments with random seed: {args.seed}")
    logging.info(f"Saved training command to file: {command_save_filename}")
    logging.info(f"Saved training arguments to file: {args_save_filename}")


def prepare_for_eval(args):
    """
    - Initialize the random seed
    - Create the output directory
    - Record the training command and arguments
    """
    set_seed(args.seed)
    if not os.path.exists(args.logging_file_path):
        os.mkdir(args.logging_file_path)
    logging_file = os.path.join(args.continue_training_path, "stdout_test.log")
    logging_stream = open(logging_file, "a")
    logging.basicConfig(logging_stream)
    logging.info(f"Run experiments with random seed: {args.seed}")


def cal_accuracy(preds, labels):
    """
    Calculate the accuracy for binary classification (only used during the training phase).
    """
    return (preds == labels).long().sum().item() / len(preds)


def label_to_language(label):
    # TODO：SNIPS数据集进来以后这个地方也要扩充
    if label == "LOC":  # conll
        label = " is a location"
    elif label == "PER":  # conll
        label = " is a person"
    elif label == "ORG":  # conll
        label = " is an organisation"
    elif label == "MISC":  # conll
        label = " is an entity"
    elif label == "GENRE":  # movie
        label = " is a genre"
    elif label == "ACTOR":  # movie
        label = " is an actor"
    elif label == "YEAR":  # movie
        label = " is a year"
    elif label == "RATINGS_AVERAGE":  # movie
        label = " is a ratings average"
    elif label == "RATING":  # movie
        label = " is a rating"
    elif label == "PLOT":  # movie
        label = " is a plot"
    elif label == "DIRECTOR":  # movie
        label = " is a director"
    elif label == "TITLE":  # movie
        label = " is a title"
    elif label == "CHARACTER":  # movie
        label = " is a character"
    elif label == "SONG":  # movie
        label = " is a song"
    elif label == "TRAILER":  # movie
        label = " is a trailer"
    elif label == "REVIEW":  # movie
        label = " is a review"
    elif label == "OPINION":
        label = " is a opinion"
    elif label == "Rating":  # restaurant
        label = " is a rating"
    elif label == "Amenity":  # restaurant
        label = " is a amenity"
    elif label == "Location":  # restaurant
        label = " is a location"
    elif label == "Restaurant_Name":  # restaurant
        label = " is a restaurant name"
    elif label == "Price":  # restaurant
        label = " is a price"
    elif label == "Hours":  # restaurant
        label = " is a hours"
    elif label == "Dish":  # restaurant
        label = " is a dish"
    elif label == "Cuisine":  # restaurant
        label = " is a cuisine"
    elif label == "arrive":  # multiwoz
        label = " is a arrive"
    elif label == "type":  # multiwoz
        label = " is a type"
    elif label == "people":  # multiwoz
        label = " is a people"
    elif label == "area":  # multiwoz
        label = " is a area"
    elif label == "price":  # multiwoz
        label = " is a price"
    elif label == "stay":  # multiwoz
        label = " is a stay"
    elif label == "leave":  # multiwoz
        label = " is a leave"
    elif label == "stars":  # multiwoz
        label = " is a stars"
    elif label == "time":  # multiwoz
        label = " is a time"
    elif label == "dest":  # multiwoz
        label = " is a dest"
    elif label == "day":  # multiwoz
        label = " is a day"
    elif label == "name":  # multiwoz
        label = " is a name"
    elif label == "food":  # multiwoz
        label = " is a food"
    elif label == "depart":  # multiwoz
        label = " is a depart"
    elif label == "movie_type":
        label = " is a movie_type"
    elif label == "album":
        label = " is a album"
    elif label == "condition_description":
        label = " is a condition_description"
    elif label == "location_name":
        label = " is a location_name"
    elif label == "track":
        label = " is a track"
    elif label == "playlist_owner":
        label = " is a playlist_owner"
    elif label == "city":
        label = " is a city"
    elif label == "timeRange":
        label = " is a timeRange"
    elif label == "cuisine":
        label = " is a cuisine"
    elif label == "facility":
        label = " is a facility"
    elif label == "current_location":
        label = " is a current_location"
    elif label == "service":
        label = " is a service"
    elif label == "artist":
        label = " is a artist"
    elif label == "party_size_description":
        label = " is a party_size_description"
    elif label == "playlist":
        label = " is a playlist"
    elif label == "condition_temperature":
        label = " is a condition_temperature"
    elif label == "state":
        label = " is a state"
    elif label == "object_select":
        label = " is a object_select"
    elif label == "music_item":
        label = " is a music_item"
    elif label == "restaurant_type":
        label = " is a restaurant_type"
    elif label == "restaurant_name":
        label = " is a restaurant_name"
    elif label == "served_dish":
        label = " is a served_dish"
    elif label == "object_name":
        label = " is a object_name"
    elif label == "rating_unit":
        label = " is a rating_unit"
    elif label == "geographic_poi":
        label = " is a geographic_poi"
    elif label == "rating_value":
        label = " is a rating_value"
    elif label == "entity_name":
        label = " is a entity_name"
    elif label == "sort":
        label = " is a sort"
    elif label == "poi":
        label = " is a poi"
    elif label == "year":
        label = " is a year"
    elif label == "movie_name":
        label = " is a movie_name"
    elif label == "genre":
        label = " is a genre"
    elif label == "party_size_number":
        label = " is a party_size_number"
    elif label == "object_part_of_series_type":
        label = " is a object_part_of_series_type"
    elif label == "best_rating":
        label = " is a best_rating"
    elif label == "spatial_relation":
        label = " is a spatial_relation"
    elif label == "country":
        label = " is a country"
    elif label == "object_type":
        label = " is a object_type"
    elif label == "object_location_type":
        label = " is a object_location_type"
    else:
        raise ValueError(f"未知label: {label}")
    return label


def evaluate(
    args,
    tokenizer,
    data_loader,
    model,
    batch_size,
    max_seq_length,
    decode_max_seq_length,
    log_file=None,
):
    if "gpt" in args.model_name_or_path:
        tokenizer.padding_side = "left"
    # evaluate by iterating data loader
    exact_match = 0
    tp, num_pred, num_true, span_tp, span_num_pred, span_num_true = 0, 0, 0, 0, 0, 0
    model.eval()
    details = []
    with tqdm.trange(math.ceil(data_loader.num_data / batch_size)) as trange_obj:
        for batch_idx in trange_obj:
            trange_obj.set_postfix(match=f"{exact_match}")
            batch = data_loader.get_batch(batch_idx, batch_size)
            batch = [
                item.cuda() if isinstance(item, torch.Tensor) else item
                for item in batch
            ]
            input_ids, input_mask, _, sentences, entities = batch
            
            generated_ids = model.generate(
                input_ids=input_ids,
                attention_mask=input_mask,
                max_length=decode_max_seq_length,
                num_beams=2,
                repetition_penalty=2.5,
                length_penalty=1.0,
                early_stopping=False,
            )
            pred_sents = [
                data_loader.tokenizer.decode(
                    g, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
                for g in generated_ids
            ]

            assert len(pred_sents) == len(entities)
            assert len(sentences) == len(entities)

            for sent, pred_sent, true_sent in zip(sentences, pred_sents, entities):
                if "gpt" in args.model_name_or_path:
                    if ";" in pred_sent:
                        pred_sent = pred_sent.split(";")[1].strip()
                pred_res = set(
                    [
                        x.strip()
                        for x in pred_sent.split(",")
                        if len(x.strip()) > 0 and "is" in x
                    ]
                )
                true_res = set(
                    [x.strip() for x in true_sent.split(",") if len(x.strip()) > 0]
                )

                tp += len(pred_res.intersection(true_res))
                is_matched = len(pred_res.intersection(true_res)) == len(true_res) and len(
                    pred_res
                ) == len(true_res)
                if not is_matched:
                    details.append(
                        {
                            "sentence": sent,
                            "true_ent": true_sent,
                            "pred_ent": " , ".join(
                                [
                                    x.strip()
                                    for x in pred_sent.split(",")
                                    if len(x.strip()) > 0
                                ]
                            ),
                        }
                    )
                exact_match += int(is_matched)
                num_pred += len(pred_res)
                num_true += len(true_res)

                span_pred_res = set(
                    [
                        x.split("is")[0].strip()
                        for x in pred_sent.split(",")
                        if len(x.strip()) > 0
                    ]
                )
                span_true_res = set(
                    [
                        x.split("is")[0].strip()
                        for x in true_sent.split(",")
                        if len(x.strip()) > 0
                    ]
                )
                span_tp += len(span_pred_res & span_true_res)
                span_num_pred += len(span_pred_res)
                span_num_true += len(span_true_res)

    p = tp / num_pred if num_pred > 0 else 0
    r = tp / num_true if num_true > 0 else 0
    span_p = span_tp / span_num_pred if span_num_pred > 0 else 0
    span_r = span_tp / span_num_true if span_num_true > 0 else 0
    metrics = {
        "precision": p,
        "recall": r,
        "f1": 2 * (p * r) / (p + r) if (p + r) > 0 else 0,
        "span_precision": span_p,
        "span_recall": span_r,
        "span_f1": 2 * (span_p * span_r) / (span_p + span_r)
        if (span_p + span_r) > 0
        else 0,
        "exact_match": exact_match / data_loader.num_data,
    }

    # log and return
    if log_file is not None:
        json.dump(metrics, open(log_file, "w"), indent=4)
        if "test" in log_file:
            json.dump(details, open(log_file[:-5] + ".detail.json", "w"), indent=4)

    return metrics

def get_sentence_ner(tokens, tags, span_only=False):
    assert len(tokens) == len(tags)

    def get_type(tag):
        return tag.split("-")[-1]

    results = []
    i = len(tags) - 1
    while i > -1:  # 从后往前找
        if tags[i] == "O":
            i -= 1
        elif tags[i][0] == "B":
            if span_only:
                results.insert(0, tokens[i])
            else:
                results.insert(0, "||".join([tokens[i], get_type(tags[i])]))
            i -= 1
        elif tags[i][0] == "I":  # 寻找B
            j = i - 1
            while (
                j > -1 and tags[j][0] == "I" and get_type(tags[i]) == get_type(tags[j])
            ):
                j -= 1
            if j == -1 or tags[j][0] == "O" or get_type(tags[i]) != get_type(tags[j]):
                i -= 1
                continue
            word = " ".join(tokens[j : i + 1])
            if span_only:
                results.insert(0, word)
            else:
                results.insert(0, "||".join([word, get_type(tags[i])]))
            i = j - 1
    return results

if __name__ == "__main__":
    main()
