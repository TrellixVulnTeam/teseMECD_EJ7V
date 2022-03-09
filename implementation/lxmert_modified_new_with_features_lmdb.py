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
import lmdb
import pickle
import io

class MyDataset(torch.utils.data.Dataset):
    def __init__(self,path):
        self.path = path
        self.size = self.getSize()
        self.env = lmdb.open(
            path, readonly=True, create=False, readahead=not False
        )
        self.txn = self.env.begin(buffers=True)
    
    def getSize(self):
        env = lmdb.open(self.path, readonly=True)
        txn = env.begin()
        count = 0
        for key, value in txn.cursor():
                count = count + 1
        env.close()
        return count
            
    def deserializeItem(self,item):
        item = pickle.loads(item)
        item['input_ids']=torch.IntTensor(item['input_ids'][0])
        item['attention_mask']=torch.IntTensor(item['attention_mask'][0])
        item['token_type_ids']=torch.IntTensor(item['token_type_ids'][0])
        item['normalized_boxes']=torch.FloatTensor(item['normalized_boxes'][0])
        item['features']=torch.FloatTensor(item['features'][0])
        return item
    
    def __getitem__(self, idx):
        item = self.txn.get(str(idx).encode())
        item = self.deserializeItem(item)
        return item

    def __len__(self):
        return self.size
    
class MyTrainer():
    def __init__(self,model,train,test):
        self.model = model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")#'cpu'
        self.train = train
        self.test = test        
    
    def train_model(self,epochs=1):
        optim = AdamW(self.model.parameters(), lr=5e-5)
        train_loader = DataLoader(self.train, batch_size=1, shuffle=True)
        for epoch in range(1):
            for item in train_loader:
                input_ids = item['input_ids'].to(self.device)
                attention_mask=item['attention_mask'].to(self.device)
                token_type_ids=item['token_type_ids'].to(self.device)
                features = item['features'].to(self.device)
                normalized_boxes = item['normalized_boxes'].to(self.device)
                label = item['label'].to(self.device)
                optim.zero_grad()
                outputs = self.model.forward(input_ids,attention_mask,token_type_ids,
                                             features,normalized_boxes,label)
                loss = outputs.loss#[0]
                loss.backward()
                optim.step()
        self.model.eval()
        self.model.save_pretrained("my_model")
        return 
        
    
class Lxmert(LxmertModel):
    def __init__(self,numb_labels=3):
        super().__init__(LxmertConfig.from_pretrained("unc-nlp/lxmert-base-uncased"))
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
    
    def forward(self,input_ids,attention_mask,token_type_ids,features,normalized_boxes,label):
        #print(inputs)
        print(input_ids.shape)
        print(attention_mask.shape)
        print(token_type_ids.shape)
        print(features.shape)
        print(normalized_boxes.shape)
        output = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            visual_feats=features,
            visual_pos=normalized_boxes,
            token_type_ids=token_type_ids,
            return_dict=True,
            output_attentions=False,
        )
        
        #print(output.pooled_output.shape)
        #aux = output.pooled_output
        #aux_mask = attention_mask
        #input_mask_expanded = aux_mask.unsqueeze(-1).expand(aux.size()).float()
        #aux = torch.sum(aux * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        
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
        img_path = data_path+'flickr30k_images/flickr30k_images/'+ train.loc[50,'Flickr30kID']#"32542645.jpg"
        question = train.loc[50,'hypothesis'] #"How many people are in the image?"
        label = train.loc[50,'gold_label']
        
        lxmert_tokenizer = LxmertTokenizer.from_pretrained("unc-nlp/lxmert-base-uncased")
        rcnn_cfg = utils.Config.from_pretrained("unc-nlp/frcnn-vg-finetuned")
        rcnn = GeneralizedRCNN.from_pretrained("unc-nlp/frcnn-vg-finetuned", config=rcnn_cfg)
        image_preprocess = Preprocess(rcnn_cfg)
        
        images, sizes, scales_yx = image_preprocess(img_path)
        
        #preprocess image
        output_dict = rcnn(
            images, 
            sizes, 
            scales_yx=scales_yx, 
            padding="max_detections",
            max_detections=rcnn_cfg.max_detections,
            return_tensors="pt"
        )
        
        #preprocess text
        inputs = lxmert_tokenizer(
            question,
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
        item = {'input_ids': inputs['input_ids'],
                'attention_mask': inputs['attention_mask'],
                'token_type_ids': inputs['token_type_ids'],
                'features':features, 
                'normalized_boxes':normalized_boxes, 
                'label':torch.LongTensor([label])}
        output = self.forward(inputs['input_ids'],inputs['attention_mask'],inputs['token_type_ids'],
                              features,normalized_boxes,torch.LongTensor([label]))
        m = torch.nn.Softmax(dim=1)
        probs = m(output.logits)
        print(img_path)
        print(question)
        print(label)
        print(probs)
        return output
        
#if __name__ == "__main__":
task = 'train'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#task = 'test'

if task =='train':
    model = Lxmert()
    model = model.to(device)
    train = MyDataset("my_train_db")
    test = MyDataset("my_test_db")
    trainer = MyTrainer(model,train, test)
    trainer.train_model()
    model.save_model("my_model2")
    output = model.run()
elif task =='test':
    model = Lxmert()
    model = model.to(model.device)
    model.load_model("my_model2")
    output = model.run()
    