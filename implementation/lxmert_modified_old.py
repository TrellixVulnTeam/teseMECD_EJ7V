import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import LxmertTokenizer, LxmertModel, LxmertForQuestionAnswering
from modeling_frcnn import GeneralizedRCNN
import utils
from processing_image import Preprocess
from transformers import Trainer

class Lxmert:
    def __init__(self):
        self.lxmert_tokenizer = LxmertTokenizer.from_pretrained("unc-nlp/lxmert-base-uncased")
        self.rcnn_cfg = utils.Config.from_pretrained("unc-nlp/frcnn-vg-finetuned")
        self.rcnn = GeneralizedRCNN.from_pretrained("unc-nlp/frcnn-vg-finetuned", config=self.rcnn_cfg)
        self.image_preprocess = Preprocess(self.rcnn_cfg)
        self.lxmert = LxmertModel.from_pretrained("unc-nlp/lxmert-base-uncased")
        #self.lxmert.classifier = torch.nn.Linear(768, 8)
        #self.lxmert.num_labels = 2

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
        
        output_vqa = self.lxmert(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            visual_feats=features,
            visual_pos=normalized_boxes,
            token_type_ids=inputs.token_type_ids,
            return_dict=True,
            output_attentions=False,
        )
        
        return output_vqa
    
    def train(self,train,test):
        trainer = Trainer(model=self, train_dataset = train, eval_dataset = test)
        #, args=training_args
        trainer.train()
    
    def run(self):
        data_path = './e-ViL/data/'
        train = pd.read_csv(data_path+'esnlive_train.csv')
        #test = pd.read_csv(data_path+'esnlive_test.csv')
        #results = pd.read_csv(data_path+'flickr30k_images/results.csv', sep ='|')
        sample = train.sample(n=100, random_state=1)
        sample_train, sample_test = train_test_split(sample, test_size=0.2)
        img_path = data_path+'flickr30k_images/flickr30k_images/'+ sample_train.loc[50,'Flickr30kID']
        question = sample_train.loc[50,'hypothesis']
        print(self.forward(question,img_path))
        self.train(sample_train,sample_test)
    
if __name__ == "__main__":
    model = Lxmert()
    output = model.run()
    print(output)