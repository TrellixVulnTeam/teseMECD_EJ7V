import json
import os
import base64

import numpy as np
import torch
from torch.utils.data import Dataset

import lmdb
import msgpack
import msgpack_numpy

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
        img_id = datum["img_id"]
        ques_id = datum["question_id"]
        ques = datum["sent"]

        # getting image features
        dump = self.txn.get(img_id.encode("utf-8"))
        nbb = self.name2nbb[img_id]
        img_dump = msgpack.loads(dump, raw=False)
        feats = img_dump["features"][:nbb, :]
        img_bb = img_dump["norm_bb"][:nbb, :]

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
            if isinstance(label, dict):  # vqa-x
                if ans in label:
                    score += label[ans]
                    correct = 1
            elif "vcr" in self.dataset.name:  # vcr
                if ans == datum["answer_choices"][label]:
                    score += 1
                    correct = 1
            else:  # esnlive
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