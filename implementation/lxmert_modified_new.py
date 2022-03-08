import torch
import pandas as pd
import numpy as np
from datasets import Dataset, load_dataset, Features
from sklearn.model_selection import train_test_split
from transformers import LxmertTokenizer, LxmertConfig, LxmertModel, LxmertForQuestionAnswering, PretrainedConfig, TrainingArguments
from modeling_frcnn import GeneralizedRCNN
import utils
from processing_image import Preprocess
from transformers import Trainer, TrainingArguments, AdamW
from torch.utils.data import DataLoader

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, questions, images, labels):
        self.questions = questions
        self.images = images
        self.labels = labels
    
    def __getitem__(self, idx):
        item = {'text':self.questions[idx],'img':self.images[idx],
                'label': self.labels[idx]}
        return item

    def __len__(self):
        return len(self.labels)
    
    
class MyTrainer():
    def __init__(self,model):
        self.train, self.test = self.read_datasets()
        self.train_dataset = MyDataset(self.train['question'].values,
                                 self.train['image'].values,
                                 self.train['label'].values)
        self.test_dataset = MyDataset(self.test['question'].values,
                                 self.test['image'].values,
                                 self.test['label'].values)
        self.model = model
    
    def read_datasets(self):
        data_path = './e-ViL/data/'
        train = pd.read_csv(data_path+'esnlive_train.csv')
        labels_encoding = {'contradiction':0,'neutral': 1,
                           'entailment':2}
        train = train[['hypothesis','Flickr30kID','gold_label']]
        train['gold_label']=train['gold_label'].apply(lambda label: labels_encoding[label])
        train['Flickr30kID'] = train['Flickr30kID'].apply(lambda x: data_path+'flickr30k_images/flickr30k_images/'+x)
        train.rename(columns={ train.columns[0]: "question", train.columns[1]: "image",
                              train.columns[2]: "label" }, inplace = True)
        sample = train.sample(n=30, random_state=1)
        sample_train, sample_test = train_test_split(sample, test_size=0.2)
        sample_train.reset_index(inplace=True,drop=True)
        sample_test.reset_index(inplace=True,drop=True)
        return sample_train, sample_test
                
    def train_model(self):
        optim = AdamW(self.model.parameters(), lr=5e-5)
        train_loader = DataLoader(self.train_dataset, batch_size=2, shuffle=True)
        for epoch in range(1):
            for item in train_loader:
                optim.zero_grad()
                outputs = model.forward(item)
                loss = outputs.loss#[0]
                loss.backward()
                optim.step()
        self.model.eval()
        return 
        
    
class Lxmert(LxmertModel):
    def __init__(self,numb_labels=3):
        super().__init__(LxmertConfig.from_pretrained("unc-nlp/lxmert-base-uncased"))
        self.lxmert_tokenizer = LxmertTokenizer.from_pretrained("unc-nlp/lxmert-base-uncased")
        self.rcnn_cfg = utils.Config.from_pretrained("unc-nlp/frcnn-vg-finetuned")
        self.rcnn = GeneralizedRCNN.from_pretrained("unc-nlp/frcnn-vg-finetuned", config=self.rcnn_cfg)
        self.image_preprocess = Preprocess(self.rcnn_cfg)
        self.config.problem_type = "single_label_classification"
        self.classification = torch.nn.Linear(self.config.hidden_size, numb_labels)
        self.num_labels = numb_labels
        
        if self.config.problem_type == "single_label_classification":
          self.loss_fct = torch.nn.CrossEntropyLoss()
          self.output_loss = lambda output,labels : self.loss_fct(output.logits.view(-1, self.num_labels), labels.view(-1)) 
        elif self.config.problem_type == "regression":
          self.loss_fct = torch.nn.MSELoss()
          if self.num_labels == 1: self.output_loss = lambda output,labels : self.loss_fct(output.logits.squeeze(), labels.squeeze())
          else: self.output_loss =  lambda output,labels : self.loss_fct(output.logits, labels)
        elif self.config.problem_type == "multi_label_classification":
          self.loss_fct = torch.nn.BCEWithLogitsLoss()
          self.output_loss = lambda output,labels : self.loss_fct(output.logits, labels)
        # don't forget to init the weights for the new layers
        self.init_weights()
        
    def forward(self,item):
        # run lxmert
        text = item['text']
        img = item['img']
        label = item['label']
        print(text)
        
        images, sizes, scales_yx = self.image_preprocess(img)
        
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
            text,
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
        
        print(inputs.input_ids.shape)
        print(inputs.attention_mask.shape)
        print(inputs.token_type_ids.shape)
        print(features.shape)
        print(normalized_boxes.shape)
        
        output = super().forward(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            visual_feats=features,
            visual_pos=normalized_boxes,
            token_type_ids=inputs.token_type_ids,
            return_dict=True,
            output_attentions=False,
        )
        
        aux = self.classification(output.pooled_output)
        output.logits = aux
        output.loss = None
        output.loss = self.output_loss(output, label)
        return output
    
    def save_model(self,path):
        torch.save(self.state_dict(), path)
        
    def load_model(self,path):
        self.load_state_dict(torch.load(path))
        self.eval()
        
    def run(self):
        data_path = './e-ViL/data/'
        train = pd.read_csv(data_path+'esnlive_train.csv')
        labels_encoding = {'contradiction':0,'neutral': 1,
                           'entailment':2}
        train['gold_label']=train['gold_label'].apply(lambda label: labels_encoding[label])
        img_path1 = data_path+'flickr30k_images/flickr30k_images/'+ train.loc[50,'Flickr30kID']#"32542645.jpg"
        question1 = train.loc[50,'hypothesis'] #"How many people are in the image?"
        label1 = train.loc[50,'gold_label']
        print('SAMPLE1')
        print(img_path1,question1,label1)
        item1 = {'text':[question1], 'img':[img_path1], 'label':torch.LongTensor([label1])}
        output = self.forward(item1)
        print(output.logits)
        m = torch.nn.Softmax(dim=1)
        probs = m(output.logits)
        print(probs)
        img_path2 = data_path+'flickr30k_images/flickr30k_images/'+ train.loc[51,'Flickr30kID']#"32542645.jpg"
        question2 = train.loc[51,'hypothesis'] #"How many people are in the image?"
        label2 = train.loc[51,'gold_label']
        print('SAMPLE2')
        print(img_path2,question2,label2)
        item2 = {'text':[question1,question2], 'img':[img_path1,img_path2], 
                'label':torch.LongTensor([label1,label2])}
        output = self.forward(item2)
        print(output.logits)
        m = torch.nn.Softmax(dim=1)
        probs = m(output.logits)
        print(probs)
        
        return output
        

#if __name__ == "__main__":
task = 'train'
#task = 'test'
if task =='train':
    model = Lxmert()
    trainer = MyTrainer(model)
    trainer.train_model()
    model.save_model("my_model")
    output = model.run()
elif task =='test':
    model = Lxmert()
    model.load_model("my_model")
    output = model.run()
