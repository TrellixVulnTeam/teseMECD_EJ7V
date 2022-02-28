import json
import os
import base64

import numpy as np
import torch
from torch.utils.data import Dataset

import lmdb
#import msgpack
import msgpack_numpy as msgpack

import collections
from torch.utils.data.dataloader import DataLoader
from time import gmtime, strftime
from transformers import get_linear_schedule_with_warmup
from tensorboardX import SummaryWriter
from tqdm import tqdm
from datasets import load_metric
import random
from my_param import args
from src.my_eval import eval_nlp_scores, input_subset, get_nlg_scores
from my_model import MyModel
from my_generation import generate_text
from src.expl_tokenization import VCRGpt2Tokenizer

DataTuple = collections.namedtuple("DataTuple", "dataset loader evaluator")

"""
Run with:
--task esnlive --train data/esnlive/esnlive_train.json --val data/esnlive/esnlive_dev.json --save_steps 5000 --output experiments/esnlive_run1/train
"""

def ctime():
    return strftime("%Y-%m-%d %H:%M:%S", gmtime())

def print_log(args, log_str):
    with open(os.path.join(args.output, "log.log"), "a") as f:
        f.write(log_str)
        f.flush()
        
def print_dict(dicto):
    out_str = ""
    for k, v in dicto.items():
        out_str += f"{k}: {v:.3f} | "
    return out_str

def pad_array(vec, pad_size):
    return np.pad(vec, [(0, pad_size - vec.shape[0]), (0, 0)], mode="constant")

def bbox_collate(batch):

    ques_id, feats, boxes, sent, target, expl, answers = [], [], [], [], [], [], []

    for example in batch:
        ques_id.append(example[0])
        feats.append(example[1])
        boxes.append(example[2])
        sent.append(example[3])
        if isinstance(example[4], int):
            target.append(example[4])
        else:
            target.append(example[4].tolist())
        expl.append(example[5])
        answers.append(example[6])

    max_len = max(map(lambda x: x.shape[0], feats))
    padded_feats = [pad_array(x, max_len) for x in feats]
    padded_boxes = [pad_array(x, max_len) for x in boxes]

    if not answers[0]:
        answers = None

    return (
        ques_id,
        torch.tensor(padded_feats).float(),
        torch.tensor(padded_boxes).float(),
        tuple(sent),
        torch.tensor(target),
        tuple(expl),
        answers,
    )

def get_data_tuple(splits: str, bs: int, shuffle=False, drop_last=False) -> DataTuple:

    dset = eViLDataset(args, splits)
    tset = eViLTorchDataset(args, dset, args.model)
    evaluator = VQAXEvaluator(dset)

    if args.task == "vqa_x":
        collate_fn = None
    else:
        collate_fn = bbox_collate

    data_loader = DataLoader(
        tset,
        batch_size=bs,
        shuffle=shuffle,
        num_workers=args.num_workers,
        drop_last=drop_last,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    return DataTuple(dataset=dset, loader=data_loader, evaluator=evaluator)


class eViLDataset:
    """
    Initialises id2datum (dict where keys are id's of each datapoint)
    Initialises an2label and label2ans, which are required throughout the code.
    """
    def __init__(self, args, splits: str):
        self.name = splits
        self.splits = splits.split(",")
        # Loading datasets
        self.data = []
        for split in self.splits:
            self.data.extend(json.load(open(split)))
        print("Load %d data from split(s) %s." % (len(self.data), self.name))
        # Convert list to dict (for evaluation)
        self.id2datum = {datum["question_id"]: datum for datum in self.data}
        # Answers
        self.ans2label = {"contradiction": 0, "neutral": 1, "entailment": 2}
        self.label2ans = {0: "contradiction", 1: "neutral", 2: "entailment"}
    
    @property
    def num_answers(self):
        return len(self.ans2label)

    def __len__(self):
        return len(self.data)
    
class eViLTorchDataset(Dataset):
    def __init__(self, args, dataset: eViLDataset, model="lxmert", max_length=50):

        super().__init__()
        self.raw_dataset = dataset
        self.model = model
        self.task = args.task
        self.FLICKR30KDB = "data/esnlive/img_db/flickr30k/feat_th0.2_max100_min10"
        self.FLICKR30KDB_NBB = "data/esnlive/img_db/flickr30k/nbb_th0.2_max100_min10.json"
        
        img_path = self.FLICKR30KDB
        nbb_path = self.FLICKR30KDB_NBB

        self.env = lmdb.open(
            img_path, readonly=True, create=False, readahead=not False
        )
        self.txn = self.env.begin(buffers=True)
        self.name2nbb = json.load(open(nbb_path))

    def __len__(self):
        return len(self.raw_dataset.data)

    def __getitem__(self, item: int):
        datum = self.raw_dataset.data[item]
        #print("DATUM\n")
        #print(datum)
        img_id = datum["img_id"]
        ques_id = datum["question_id"]
        ques = datum["sent"]

        # getting image features
        #print(img_id.encode("utf-8"))
        dump = self.txn.get(img_id.encode("utf-8"))
        #print(dump)
        nbb = self.name2nbb[img_id]
        img_dump = msgpack.loads(dump, raw=False)
        #print(nbb)
        #print(img_dump["features"][b'data'])
        #print(img_dump.keys())
        #print(img_dump["features"].keys())
        #print(img_dump["norm_bb"].keys())
        feats = img_dump["features"][:nbb, :]
        img_bb = img_dump["norm_bb"][:nbb, :]
        #feats = img_dump["features"][b'data'][:nbb,:]
        #img_bb = img_dump["norm_bb"][b'data'][:nbb,:]
        

        # get box to same format than used by code's authors
        boxes = np.zeros((img_bb.shape[0], 7), dtype="float32")
        boxes[:, :-1] = img_bb[:, :]
        boxes[:, 4] = img_bb[:, 5]
        boxes[:, 5] = img_bb[:, 4]
        boxes[:, 4] = img_bb[:, 5]
        boxes[:, 6] = boxes[:, 4] * boxes[:, 5]

        if "label" in datum:
            label = datum["label"]
            target = self.raw_dataset.ans2label[label]
            if "explanation" in datum:
                # get multiple expl for validatin of vqa-x, else just one
                expl = datum["explanation"][0]
                if "answer_choices" in datum:  # required for conditioning explanations
                    answers = datum["answer_choices"]
                else:
                    answers = 1
                return ques_id, feats, boxes, ques, target, expl, answers
            else:
                return ques_id, feats, boxes, ques, target
        else:
            return ques_id, feats, boxes, ques

    def _decodeIMG(self, img_info):
        img_h = int(img_info[1])
        img_w = int(img_info[2])
        boxes = img_info[-2]
        boxes = np.frombuffer(base64.b64decode(boxes), dtype=np.float32)
        boxes = boxes.reshape(36, 4)
        boxes.setflags(write=False)
        feats = img_info[-1]
        feats = np.frombuffer(base64.b64decode(feats), dtype=np.float32)
        feats = feats.reshape(36, -1)
        feats.setflags(write=False)
        return [img_h, img_w, boxes, feats]
    
class VQAXEvaluator:
    def __init__(self, dataset: eViLDataset):
        self.dataset = dataset

    def evaluate(self, quesid2ans: dict):
        score = 0.0
        correct_idx = []
        for quesid, ans in quesid2ans.items():
            datum = self.dataset.id2datum[quesid]
            label = datum["label"]
            correct = 0
            if ans == label:
                score += 1
                correct = 1
            correct_idx.append(correct)

        return score / len(quesid2ans), correct_idx

    def dump_result(self, quesid2ans: dict, path):
        """
        Dump results to a json file, which could be submitted to the VQA online evaluation.
        VQA json file submission requirement:
            results = [result]
            result = {
                "question_id": int,
                "answer": str
            }

        :param quesid2ans: dict of quesid --> ans
        :param path: The desired path of saved file.
        """
        with open(path, "w") as f:
            result = []
            for ques_id, ans in quesid2ans.items():
                result.append({"question_id": ques_id, "answer": ans})
            json.dump(result, f, indent=4, sort_keys=True)

def weighted_loss(task_loss, expl_loss, loss_weights, cweight):

    # get loss after dwa weighting
    l_c = loss_weights["task"] * task_loss
    l_e = loss_weights["expl"] * expl_loss

    # makes sure sum of losses remains the same and ratio changes cweight-fold
    w_e = (float(l_c) + float(l_e)) / (cweight * float(l_c) + float(l_e))
    w_c = cweight * w_e

    return w_c * l_c + w_e * l_e

def write_items(items, output_file):
    with open(output_file, "w") as f:
        for item in items:
            f.write(str(item) + "\n")
    f.close()
    
def random_print_samples(sent, label, generated_explanations, label2ans):
    """
    Prints a random subset of generated explanations.
    """
    if np.random.choice(np.arange(0, 2), p=[1 - len(sent) / 100, len(sent) / 100]):
        idx = random.randrange(len(sent))
        question_ex = sent[idx]
        label_ex = label[idx]
        if isinstance(label2ans[0], list):
            answer_ex = label2ans[idx][label_ex]
        else:
            answer_ex = label2ans[label_ex]
        explanation_ex = generated_explanations[idx]

        print(
            f"\n********** EVAL EXAMPLE ********** || Question: {question_ex} | Answer: {answer_ex} | Explanation: {explanation_ex}"
        )
    
class VQA:
    def __init__(self):

        self.train_type = args.train_type
        self.device = torch.device(args.device)

        # Dataloaders for train and val set
        if not args.test:
            self.valid_tuple = get_data_tuple(
                args.valid, bs=args.batch_size, shuffle=False, drop_last=False
            )
            self.train_tuple = get_data_tuple(
                args.train, bs=args.batch_size, shuffle=True, drop_last=True
            )
            num_answers = self.train_tuple.dataset.num_answers
            file_name = args.train
            log_str = f"\n{ctime()} || Loaded train set of size {len(self.train_tuple[0])} and val set of size {len(self.valid_tuple[0])}."
        else:
            self.test_tuple = get_data_tuple(
                args.test, bs=args.batch_size, shuffle=False, drop_last=False
            )
            num_answers = self.test_tuple.dataset.num_answers
            file_name = args.test
            log_str = (
                f"\n{ctime()} || Loaded test set of size {len(self.test_tuple[0])}."
            )

        # get dataset name
        self.dtype = args.task

        # Model
        self.model = MyModel(self.train_type, num_answers, self.dtype, args.model)

        # Load pre-trained weights
        if self.train_type == "expl" and args.bb_path is not None:
            self.model.load_state_dict(torch.load(args.bb_path))
            # freeze backbone
            for p, n in self.model.named_parameters():
                if "decoder.model.transformer" not in p:
                    n.requires_grad = False
        elif args.load_pretrained is not None:
            self.model.encoder.load(args.load_pretrained)

        self.model = self.model.to(self.device)

        # Loss and Optimizer
        if not args.test:
            if self.dtype == "vqa_x":
                self.loss_func = torch.nn.BCEWithLogitsLoss()
            else:
                self.loss_func = torch.nn.CrossEntropyLoss()

            batch_per_epoch = len(self.train_tuple.loader) / args.grad_accum
            t_total = int(batch_per_epoch * args.epochs)

            if "bert" in args.optim:
                print("BertAdam Total Iters: %d" % t_total)
                from src.optimization import BertAdam

                self.optim = BertAdam(
                    list(self.model.parameters()),
                    lr=args.lr,
                    warmup=0.1,
                    t_total=t_total,
                )
            else:
                self.optim = args.optimizer(self.model.parameters(), args.lr)
                self.scheduler = get_linear_schedule_with_warmup(
                    self.optim,
                    num_warmup_steps=args.warmup_steps,
                    num_training_steps=t_total,
                )
        self.grad_accum = args.grad_accum

        # Output Directory
        self.output = args.output
        self.save_steps = args.save_steps
        os.makedirs(self.output, exist_ok=True)

        # print logs
        log_str += f"\n{ctime()} || Model loaded. Batch size {args.batch_size*args.grad_accum} | lr {args.lr} | task: {self.dtype} | type: {self.train_type}."
        print_log(args, log_str)

    def train(self, train_tuple, eval_tuple):

        tb_writer = SummaryWriter(self.output)

        dset, loader, evaluator = train_tuple
        iter_wrapper = (
            (lambda x: tqdm(x, total=len(loader))) if args.tqdm else (lambda x: x)
        )

        # logger initialisations
        best_task = 0.0  # this refers to the model with the best S_T score
        best_expl = 0.0  # this refers to the model with the best S_E score
        best_global = 0.0  # this refers to the model with the best S_O score
        prev_losses = [[1], [1]]
        prev_task, prev_expl = 0, 0
        global_step = 0
        t_loss, tt_loss, te_loss = 0, 0, 0
        step_per_eval = 0

        for epoch in range(args.epochs):
            quesid2ans = {}
            for i, (
                ques_id,
                feats,
                boxes,
                sent,
                target,
                expl,
                answer_choices,
            ) in iter_wrapper(enumerate(loader)):

                self.model.train()
                self.optim.zero_grad()

                expl_gt = target

                model_dict = dset.label2ans

                logit, output, _, _, _ = self.model(
                    feats.to(self.device),
                    boxes.to(self.device),
                    sent,
                    expl,
                    answer_choices,
                    model_dict,
                    expl_gt,
                )

                loss_multiplier = 1

                if self.train_type == "all":
                    task_loss = (
                        self.loss_func(logit, target.to(self.device)) * loss_multiplier
                    )
                    expl_loss = output[0]
                    # loss_weights = dwa(prev_losses, temp=args.temperature)
                    loss_weights = {"task": 1, "expl": 1}
                    # loss = loss_weights['task']*task_loss + loss_weights['expl']*expl_loss
                    loss = weighted_loss(
                        task_loss, expl_loss, loss_weights, args.classifier_weight
                    )
                    loss /= self.grad_accum

                    prev_task += float(task_loss)
                    prev_expl += float(expl_loss)

                    # record loss for every 1024 datapoints
                    if (i + 1) % int((1024 / args.batch_size)) == 0:
                        prev_losses[0].append(prev_task / (1024 / args.batch_size))
                        prev_losses[1].append(prev_expl / (1024 / args.batch_size))
                        prev_task, prev_expl = 0, 0

                elif self.train_type == "bb":
                    loss = (
                        self.loss_func(logit, target.to(self.device)) * loss_multiplier
                    )
                    loss /= self.grad_accum
                    task_loss = float(loss)
                    expl_loss = 0

                elif self.train_type == "expl":
                    loss = output[0]
                    loss /= self.grad_accum
                    task_loss = 0
                    expl_loss = float(loss)

                loss.backward()

                score, label = logit.max(1)
                if not isinstance(ques_id, list):
                    ques_id = ques_id.cpu().numpy()


                for qid, l in zip(ques_id, label.cpu().numpy()):
                    ans = dset.label2ans[l]
                    quesid2ans[qid] = ans

                t_loss += float(loss) * self.grad_accum
                tt_loss += float(task_loss)
                te_loss += float(expl_loss)
                step_per_eval += 1

                # global step
                # grad accum snippet: https://gist.github.com/thomwolf/ac7a7da6b1888c2eeac8ac8b9b05d3d3
                if (i + 1) % self.grad_accum == 0:

                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                    self.optim.step()
                    if args.optim != "bert":
                        self.scheduler.step()  # Update learning rate schedule

                    # logging
                    tb_writer.add_scalar("task loss", task_loss, global_step)
                    tb_writer.add_scalar("explanation loss", expl_loss, global_step)
                    tb_writer.add_scalar(
                        "total loss", float(loss) * self.grad_accum, global_step
                    )
                    if self.train_type == "all":
                        tb_writer.add_scalar(
                            "task weight", loss_weights["task"], global_step
                        )
                        tb_writer.add_scalar(
                            "explanation weight", loss_weights["expl"], global_step
                        )

                    global_step += 1

                    # do eval
                    if self.save_steps > 0 and global_step % self.save_steps == 0:
                        log_str = f"\n\n{ctime()} || EVALUATION TIME"
                        log_str += f"\nEpoch-step {epoch}-{global_step}: Loss {t_loss/step_per_eval:.2f} | Task loss {tt_loss/step_per_eval:.2f} | Expl loss {te_loss/step_per_eval:.2f} | Train acc {evaluator.evaluate(quesid2ans)[0]:.2f}"
                        print_log(args, log_str)
                        t_loss, tt_loss, te_loss = 0, 0, 0
                        step_per_eval = 0

                        if self.valid_tuple is not None:  # Do Validation
                            valid_score, valid_perplexity, nlg_scores = self.evaluate(
                                eval_tuple
                            )

                            # no explanations generated
                            if not nlg_scores:

                                if valid_score > best_task:
                                    best_task = valid_score
                                    self.save("best_task")

                                log_str = f"\nEpoch-step {epoch}-{global_step}: Valid Score: {valid_score:.3f} | Best Valid Score: {best_task:.3f}"
                                tb_writer.add_scalar(
                                    "valid_task_score", valid_score * 100.0, global_step
                                )
                                tb_writer.add_scalar(
                                    "valid_expl_perplexity",
                                    valid_perplexity * 100.0,
                                    global_step,
                                )
                                print_log(args, log_str)
                                continue

                            if valid_score > best_task:
                                best_task = valid_score
                                self.save("best_task")

                            if self.train_type == "bb":
                                nlg_avg = 0
                                global_score = 0
                                valid_perplexity = 0
                            else:
                                global_score = nlg_scores["global_score"]
                                if global_score > best_global:
                                    best_global = global_score
                                    self.save("best_global")

                                nlg_avg = nlg_scores["avg_all"]
                                if nlg_avg > best_expl:
                                    best_expl = nlg_avg
                                    self.save("best_expl")

                            log_str = f"\nEpoch-step {epoch}-{global_step}: Valid Score: {valid_score:.3f} | NLG average: {nlg_avg:.3f} | Global score: {global_score:.3f}"
                            log_str += f"\nEpoch-step {epoch}-{global_step}: Best Valid Score: {best_task:.3f} | Best NLG: {best_expl:.3f} | Best overall: {best_global:.3f}"

                            tb_writer.add_scalar(
                                "valid_task_score", valid_score * 100.0, global_step
                            )
                            tb_writer.add_scalar(
                                "valid_expl_perplexity",
                                valid_perplexity * 100.0,
                                global_step,
                            )

                            if nlg_scores:
                                log_str += f"\nEpoch-step {epoch}-{global_step}: {print_dict(nlg_scores)}"
                                for k, v in nlg_scores.items():
                                    tb_writer.add_scalar(k, v, global_step)

                        print(log_str, end="")

                        print_log(args, log_str)

                        tb_writer.flush()

        self.save("LAST")
        tb_writer.close()

    def predict(self, train_type, eval_tuple: DataTuple, dump=None, gen_dump=None):
        """
        Predict the answers to questions in a data split.

        :param eval_tuple: The data tuple to be evaluated.
        :param dump: The path of saved file to dump results.
        :return: A dict of question_id to answer.
        """

        self.model.eval()
        dset, loader, evaluator = eval_tuple
        quesid2ans = {}
        expl_loss = 0.0
        nb_eval_steps = 0
        generated_explanations = None
        test_output = []

        if "bb" not in train_type:
            # initialisations for NL evaluation
            try:
                bert_metric = load_metric(
                    "bertscore",
                    experiment_id=str(random.randrange(999999)),
                    device=self.device,
                )
            except:
                bert_metric = None
            all_generated_explanations = []
            all_gt_expls = []
            tokenizer = VCRGpt2Tokenizer.from_pretrained("gpt2")
            gen_model = self.model.decoder.model.to(self.device)

        for i, datum_tuple in enumerate(loader):
            ques_id, feats, boxes, sent, label, expl, answers = datum_tuple

            if args.gt_cond:
                gt = label
            else:
                gt = None

            model_dict = dset.label2ans

            triple_expl = None

            with torch.no_grad():
                feats, boxes = feats.to(self.device), boxes.to(self.device)
                (
                    logit,
                    expl_output,
                    input_ids,
                    token_type_ids,
                    visual_representations,
                ) = self.model(feats, boxes, sent, expl, answers, model_dict, gt)


                correct_indices = (
                    torch.where(label.to(self.device) == torch.argmax(logit, 1))[0]
                    .detach()
                    .cpu()
                )
                if args.gt_cond:
                    correct_indices = torch.range(0, label.size(0) - 1, dtype=int)

                # populate quesid2ans (where ans is predicted ans)
                if not isinstance(ques_id, list):
                    ques_id = ques_id.cpu().numpy()
                score, label = logit.max(1)
                
                for qid, l in zip(ques_id, label.cpu().numpy()):
                    ans = dset.label2ans[l]
                    quesid2ans[qid] = ans

                # generate and evaluate explanations
                get_gen_expl = 0
                if "bb" not in train_type:
                    expl_loss += expl_output[0].mean().item()

                    # only evaluate random subset during validation to save time
                    if args.test:
                        get_gen_expl = 1
                    else:
                        get_gen_expl = np.random.choice(
                            np.arange(0, 2), p=[1 - args.prob_eval, args.prob_eval]
                        )

                    # get subset where label was predicted correctly
                    (
                        input_ids,
                        token_type_ids,
                        visual_representations,
                        expl,
                        triple_expl,
                    ) = input_subset(
                        correct_indices,
                        input_ids,
                        token_type_ids,
                        visual_representations,
                        expl,
                        triple_expl,
                        self.device,
                    )
                    generated_explanations = None

                    if input_ids.shape[0] != 0:  # if not all predictions were wrong
                        if get_gen_expl:
                            generated_explanations = generate_text(
                                gen_model,
                                tokenizer,
                                input_ids,
                                token_type_ids,
                                visual_representations,
                                max_rationale_length=51,
                            )

                            
                            try:
                                bert_metric.add_batch(
                                    predictions=generated_explanations,
                                    references=expl,
                                )
                            except:
                                print("BertScore failed")
                            all_gt_expls.extend(expl)

                            all_generated_explanations.extend(generated_explanations)

                            # printing examples during eval
                            if not args.test:
                                labels = [label[i].item() for i in correct_indices]
                                random_print_samples(
                                    [sent[i] for i in correct_indices],
                                    labels,
                                    generated_explanations,
                                    model_dict,
                                )

                gen_expl_all = len(ques_id) * ["None"]
                if generated_explanations:
                    for ci, gen_expl in zip(correct_indices, generated_explanations):
                        gen_expl_all[ci] = gen_expl

                # write explanations to file
                if gen_dump:
                    for idx, (qid, gen_expl) in enumerate(
                        zip(list(ques_id), gen_expl_all)
                    ):
                        input_record = {}

                        input_record["question_id"] = str(qid)
                        input_record["question"] = dset.id2datum[qid]["sent"]
                        input_record["generated_explanation"] = gen_expl
                        input_record["correct_explanations"] = dset.id2datum[qid][
                            "explanation"
                        ]
                        input_record["prediction"] = quesid2ans[qid]
                        input_record["gt"] = dset.id2datum[qid]["label"]
                        input_record["img_id"] = str(qid)[:-5]
                        if idx in list(correct_indices.numpy()):
                            input_record["correct"] = 1
                        else:
                            input_record["correct"] = 0

                        test_output.append(input_record)

            nb_eval_steps += 1

        valid_score, correct_idx = eval_tuple.evaluator.evaluate(quesid2ans)
        nlg_weight = correct_idx.count(1) / len(
            correct_idx
        )  # because for vqa-x we also take half-correct answers

        # getting perplexity
        expl_loss = expl_loss / nb_eval_steps
        perplexity = torch.exp(torch.tensor(expl_loss)).item()

        if "bb" not in train_type and len(all_generated_explanations) != 0:

            # getting NLG metrics
            nlg_global_scores = get_nlg_scores(
                self.dtype,
                all_generated_explanations,
                all_gt_expls,
                bert_metric,
                self.device,
            )
            nlg_global_scores["global_score"] = (
                nlg_global_scores["avg_all"] * nlg_weight
            )
            if not nlg_global_scores["global_score"]:
                nlg_global_scores["global_score"] = 0

            if gen_dump is not None:
                scores_to_print = nlg_global_scores
                scores_to_print["task_score"] = valid_score
                write_items(
                    [json.dumps(r) for r in ["scores", scores_to_print]],
                    os.path.join(args.output, "scores.json"),
                )
                write_items(
                    [json.dumps(r) for r in test_output],
                    os.path.join(args.output, "gen_test.json"),
                )

            return valid_score, perplexity, nlg_global_scores
        else:
            scores_to_print = {"task_score": valid_score}
            print("Task Score: ", valid_score)
            write_items(
                [json.dumps(r) for r in ["scores", scores_to_print]],
                os.path.join(args.output, "scores.json"),
            )
            return valid_score, perplexity, None

    def evaluate(self, eval_tuple: DataTuple, dump=None):
        """Evaluate all data in data_tuple."""
        valid_score, expl_perplexity, nlg_global_scores = self.predict(
            self.train_type, eval_tuple, dump
        )
        return valid_score, expl_perplexity, nlg_global_scores

    @staticmethod
    def oracle_score(data_tuple):
        """
        Purpose:
        """
        dset, loader, evaluator = data_tuple
        quesid2ans = {}
        for i, (ques_id, feats, boxes, sent, target) in enumerate(loader):
            _, label = target.max(1)
            for qid, l in zip(ques_id, label.cpu().numpy()):
                ans = dset.label2ans[l]
                quesid2ans[qid.item()] = ans
        return evaluator.evaluate(quesid2ans)

    def save(self, name):
        torch.save(self.model.state_dict(), os.path.join(self.output, "%s.pth" % name))

    def load(self, path):
        print("Load model from %s" % path)
        state_dict = torch.load("%s.pth" % path, map_location=torch.device("cpu"))
        self.model.load_state_dict(state_dict, strict=False)
        self.model = self.model.to(self.device)

"""    
args = None
splits = 16
dset = eViLDataset(args, splits)
tset = eViLTorchDataset(args, dset, args.model)
"""

# logging
if not os.path.exists(args.output):
    os.makedirs(args.output)
print_log(args, "\n" + str(args) + "\n")
tb_path = os.path.join(os.getcwd(), args.output)
log_str = f"\ntensorboard dev upload --logdir {tb_path} --name ug-tt_{args.train_type}-bs{args.batch_size*args.grad_accum}-lr{args.lr}-t{args.temperature}"
log_str += f"\n Device: {torch.cuda.current_device()}"
log_str += f"\n Process ID: {os.getpid()}"
print_log(args, log_str)

# Build Class
vqa = VQA()

# Load VQA model weights
if args.load_trained is not None:
    vqa.load(args.load_trained)

# Test or Train
if args.test:
    valid_score, perplexity, nlg_global_scores = vqa.predict(
        args.train_type,
        vqa.test_tuple,
        dump=os.path.join(args.output, "test_predict.json"),
        gen_dump=os.path.join(args.output, "gen_output.json"),
    )
else:
    print("Splits in Train data:", vqa.train_tuple.dataset.splits)
    if vqa.valid_tuple is not None:
        print("Splits in Valid data:", vqa.valid_tuple.dataset.splits)
        # print("Valid Oracle: %0.2f" % (vqa.oracle_score(vqa.valid_tuple) * 100))
    else:
        print("DO NOT USE VALIDATION")
    vqa.train(vqa.train_tuple, vqa.valid_tuple)


