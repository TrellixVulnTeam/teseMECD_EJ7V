#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#from PIL import Image
#import requests
#requests.__version__
#!pip install requests==2.27.1
#get_ipython().system('pip install transformers')
#get_ipython().system('pip install wget')


# In[ ]:


#from google.colab import drive
#drive.mount('/content/drive')


# In[ ]:


#import sys
#import os
#sys.path.append(os.path.abspath('/content/drive/MyDrive/teses/tese_MECD/implementation'))


# In[ ]:


#get_ipython().run_line_magic('cd', "'/content/drive/MyDrive/teses/tese_MECD/implementation'")


# In[ ]:


#url = "./data/flickr30k_images/flickr30k_images/5897297135.jpg"
#os.path.isfile(url)
#os.path.isfile('./data/flickr30k_images/flickr30k_images/4852389235.jpg')
#image = Image.open(requests.get(url, stream=True).raw).convert("RGB")


# In[15]:


import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import LxmertTokenizer, LxmertConfig, LxmertModel
from modeling_frcnn import GeneralizedRCNN
import utils
from processing_image import Preprocess
from torch.optim import AdamW
from torch.utils.data import DataLoader


# In[16]:


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, hypothesis, images, labels):
        self.hypothesis = hypothesis
        self.images = images
        self.labels = labels

    
    def __getitem__(self, idx):
        item = {'text':torch.Tensor(self.hypothesis[idx]),'img':torch.Tensor(self.images[idx]),
                'label': self.labels[idx]}
        return item

    def __len__(self):
        return len(self.labels)


# In[30]:


class MyTrainer():
    def __init__(self,model):
        self.train = self.read_dataset(data_path ='../e-ViL/data/', dataset_path='esnlive_train.csv',
                                       img_path='flickr30k_images/flickr30k_images/')
        self.test = self.read_dataset(data_path ='../e-ViL/data/', dataset_path='esnlive_test.csv',
                                       img_path='flickr30k_images/flickr30k_images/')
        self.dev = self.read_dataset(data_path ='../e-ViL/data/', dataset_path='esnlive_dev.csv',
                                       img_path='flickr30k_images/flickr30k_images/')
        self.train_dataset = MyDataset(self.train['hypothesis'].values,
                                 self.train['image'].values,
                                 self.train['label'].values)
        self.test_dataset = MyDataset(self.test['hypothesis'].values,
                                 self.test['image'].values,
                                 self.test['label'].values)
        self.dev_dataset = MyDataset(self.dev['hypothesis'].values,
                                 self.dev['image'].values,
                                 self.dev['label'].values)
        self.model = model
      
    def read_dataset(self,data_path=None,dataset_path=None,img_path=None):
        data = pd.read_csv(data_path+dataset_path)
        labels_encoding = {'contradiction':0,'neutral': 1,
                           'entailment':2}
        data = data[['hypothesis','Flickr30kID','gold_label']]
        data['gold_label']=data['gold_label'].apply(lambda label: labels_encoding[label])
        data['Flickr30kID'] = data['Flickr30kID'].apply(lambda x: data_path+img_path+x)
        data.rename(columns={ data.columns[0]: "hypothesis", data.columns[1]: "image",
                              data.columns[2]: "label" }, inplace = True)
        return data
                
    def train_model(self,lr = 1e-2, batch_size = 8, epochs = 10):
        optim = AdamW(self.model.parameters(), lr=lr)
        train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        for epoch in range(epochs):
            for item in train_loader:
                print(item)
                optim.zero_grad()
                outputs = model.forward(item)
                loss = outputs.loss
                loss.backward()
                optim.step()
                break
            break
        self.model.eval()
        return


# In[37]:


class Lxmert(LxmertModel):
    def __init__(self,device='cpu',numb_labels=3):
        super().__init__(LxmertConfig.from_pretrained("unc-nlp/lxmert-base-uncased"))
        self.model_device = device
        self.to(self.model_device)
        self.config.problem_type = "single_label_classification"
        self.num_labels = numb_labels
        self.classification = torch.nn.Linear(self.config.hidden_size, self.num_labels)
        self.lxmert_tokenizer = LxmertTokenizer.from_pretrained("unc-nlp/lxmert-base-uncased")
        self.rcnn_cfg = utils.Config.from_pretrained("unc-nlp/frcnn-vg-finetuned")
        self.rcnn_cfg.MODEL.DEVICE = self.model_device
        self.rcnn = GeneralizedRCNN.from_pretrained("unc-nlp/frcnn-vg-finetuned", config=self.rcnn_cfg)
        self.image_preprocess = Preprocess(self.rcnn_cfg)
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
    
    def get_visual_features(self,img):
        #preprocess image
        images, sizes, scales_yx = self.image_preprocess(img)
        output_dict = self.rcnn(
            images, 
            sizes, 
            scales_yx=scales_yx, 
            padding="max_detections",
            max_detections=self.rcnn_cfg.max_detections,
            return_tensors="pt"
        )
        
        #Very important that the boxes are normalized
        normalized_boxes = output_dict.get("normalized_boxes")
        features = output_dict.get("roi_features")
        return normalized_boxes, features
    
    def get_text_features(self,text): 
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
        return inputs
    
    def forward(self,item):
        text = item['text']
        img = item['img']
        label = item['label']
        print(np.shape(text))
        print(np.shape(img))
        print(np.shape(label))
        inputs = self.get_text_features(text)
        normalized_boxes, features = self.get_visual_features(img)
        inputs = inputs.to(self.model_device)
        normalized_boxes = normalized_boxes.to(self.model_device)
        features = features.to(self.model_device)
        label = label.to(self.model_device)
        
        print(normalized_boxes.shape)
        print(features.shape)
        print(inputs.input_ids.shape)
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
        output.loss = self.output_loss(output, label)
        return output
    
    def save_model(self,path):
        torch.save(self.state_dict(), path)
        
    def load_model(self,path):
        self.load_state_dict(torch.load(path))
        self.eval()
    
    def predict(self,X):
      """
      X (n_examples x n_features)
      """
      scores = model(X)  # (n_examples x n_classes)
      predicted_labels = scores.argmax(dim=-1)  # (n_examples)
      return predicted_labels

    def evaluate(self, X, y):
      """
      X (n_examples x n_features)
      y (n_examples): gold labels
      """
      self.eval()
      y_hat = self.predict(X)
      n_correct = (y == y_hat).sum().item()
      n_possible = float(y.shape[0])
      self.train()
      return n_correct / n_possible

    def run_example(self,dataset,trainer):
        img_path1 = dataset.loc[50,'image']
        text1 = dataset.loc[50,'hypothesis'] #"How many people are in the image?"
        label1 = torch.LongTensor([dataset.loc[50,'label']])
        print('SAMPLE1')
        print(img_path1,text1,label1)
        item1 = {'text':[text1], 'img':[img_path1], 'label':label1}
        inputs = trainer.get_text_features(text1)
        normalized_boxes, features = trainer.get_visual_features(img_path1)        
        output = self.forward(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            features=features,
            normalized_boxes=normalized_boxes,
            token_type_ids=inputs.token_type_ids,
            label = label1
        )
        print(output.logits)
        m = torch.nn.Softmax(dim=1)
        probs = m(output.logits)
        print(probs)
        return output


# In[38]:


device = "cpu"
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
task = 'train'
#device


# In[39]:


if task =='train':
    model = Lxmert(device)
    trainer = MyTrainer(model)
    print("-----Training Model-----")
    trainer.train_model(lr = 1e-2, batch_size = 8, epochs = 10)
    model.save_model("/content/drive/MyDrive/teses/tese_MECD/implementation/my_model")
    print('----Training finished-----')
elif task =='test':
    model = Lxmert()
    model.load_model("/content/drive/MyDrive/teses/tese_MECD/implementation/my_model")
    trainer = MyTrainer(model,device=device)
    output = model.run_example(trainer.test,trainer)


# In[40]:


#%reset
#import gc
#gc.collect()
import gc
model = None
trainer = None
gc.collect()


# In[ ]:




