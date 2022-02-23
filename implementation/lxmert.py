import torch
import pandas as pd
from transformers import LxmertTokenizer, LxmertModel, LxmertForQuestionAnswering
from modeling_frcnn import GeneralizedRCNN
import utils
from processing_image import Preprocess
from sklearn.model_selection import train_test_split

class Lxmert:
    def __init__(self):
        self.lxmert_tokenizer = LxmertTokenizer.from_pretrained("unc-nlp/lxmert-base-uncased")
        self.rcnn_cfg = utils.Config.from_pretrained("unc-nlp/frcnn-vg-finetuned")
        self.rcnn = GeneralizedRCNN.from_pretrained("unc-nlp/frcnn-vg-finetuned", config=self.rcnn_cfg)
        self.image_preprocess = Preprocess(self.rcnn_cfg)
        self.lxmert_gqa = LxmertForQuestionAnswering.from_pretrained("unc-nlp/lxmert-gqa-uncased")
        GQA_URL = "https://raw.githubusercontent.com/airsplay/lxmert/master/data/gqa/trainval_label2ans.json"
        self.gqa_answers = utils.get_data(GQA_URL)
        self.lxmert_vqa = LxmertForQuestionAnswering.from_pretrained("unc-nlp/lxmert-vqa-uncased")
        VQA_URL = "https://raw.githubusercontent.com/airsplay/lxmert/master/data/vqa/trainval_label2ans.json"
        self.vqa_answers = utils.get_data(VQA_URL)
        

    def forward(self,text,img):
        # run lxmert
        test_question = [text]
        URL = img
        
        images, sizes, scales_yx = self.image_preprocess(URL)
        
        #preprocess image
        output_dict = self.rcnn(
            images, 
            sizes, 
            scales_yx=scales_yx, 
            padding="max_detections",
            max_detections=self.rcnn_cfg.max_detections,
            return_tensors="pt"
        )
        
        #preprocess text
        inputs = self.lxmert_tokenizer(
            test_question,
            padding="max_length",
            max_length=20,
            truncation=True,
            return_token_type_ids=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt"
        )
        
        #Very important that the boxes are normalized
        normalized_boxes = output_dict.get("normalized_boxes")
        features = output_dict.get("roi_features")

        # run lxmert(s)
        output_gqa = self.lxmert_gqa(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            visual_feats=features,
            visual_pos=normalized_boxes,
            token_type_ids=inputs.token_type_ids,
            return_dict=True,
            output_attentions=False,
        )
        
        output_vqa = self.lxmert_vqa(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            visual_feats=features,
            visual_pos=normalized_boxes,
            token_type_ids=inputs.token_type_ids,
            return_dict=True,
            output_attentions=False,
        )
        
        # get prediction
        pred_vqa = output_vqa["question_answering_score"].argmax(-1)
        pred_gqa = output_gqa["question_answering_score"].argmax(-1)
        
        return self.gqa_answers[pred_gqa], self.vqa_answers[pred_vqa]
    
    def train(self):
        data_path = './e-ViL/data/'
        train = pd.read_csv(data_path+'esnlive_train.csv')
        test = pd.read_csv(data_path+'esnlive_test.csv')
        results = pd.read_csv(data_path+'flickr30k_images/results.csv', sep ='|')
        sample = train.sample(n=100, random_state=1)
        sample_train, sample_test = train_test_split(sample, test_size=0.2)
        sample_train = sample_train.reset_index()
        sample_test = sample_test.reset_index()
        img_path = data_path+'flickr30k_images/flickr30k_images/'+ sample_train.loc[50,'Flickr30kID']#"32542645.jpg"
        question = sample_train.loc[50,'hypothesis'] #"How many people are in the image?"
        print(question)
        return self.forward(question,img_path)
    
if __name__ == "__main__":
    model = Lxmert()
    output = model.train()
    print(output)