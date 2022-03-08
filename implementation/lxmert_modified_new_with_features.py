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
    def __init__(self,inputs,features, normalized_boxes,labels):
        self.inputs = inputs
        self.features = features
        self.normalized_boxes = normalized_boxes
        self.labels = labels
    
    def __getitem__(self, idx):
        item = {'input_ids': self.inputs[idx]['input_ids'][0],
                'attention_mask': self.inputs[idx]['attention_mask'][0],
                'token_type_ids': self.inputs[idx]['token_type_ids'][0],
                'features':self.features[idx][0],
                'normalized_boxes': self.normalized_boxes[idx][0],
                'label': self.labels[idx]}
        return item

    def __len__(self):
        return len(self.labels)
    
class MyDataLoader():
    def __init__(self):
        self.lxmert_tokenizer = LxmertTokenizer.from_pretrained("unc-nlp/lxmert-base-uncased")
        self.rcnn_cfg = utils.Config.from_pretrained("unc-nlp/frcnn-vg-finetuned")
        self.rcnn = GeneralizedRCNN.from_pretrained("unc-nlp/frcnn-vg-finetuned", config=self.rcnn_cfg)
        self.image_preprocess = Preprocess(self.rcnn_cfg)
        self.train, self.test = self.read_datasets()
        self.train_processed = self.process_dataset(self.train)
        self.test_processed = self.process_dataset(self.test)
        self.train_dataset = MyDataset(self.train_processed['inputs'].values,
                                 self.train_processed['features'].values,
                                 self.train_processed['normalized_boxes'].values,
                                 self.train_processed['label'].values)
        self.test_dataset = MyDataset(self.test_processed['inputs'].values,
                                 self.test_processed['features'].values,
                                 self.test_processed['normalized_boxes'].values,
                                 self.test_processed['label'].values)
        return
    
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
        sample = train.sample(n=10, random_state=1)
        sample_train, sample_test = train_test_split(sample, test_size=0.2)
        sample_train.reset_index(inplace=True,drop=True)
        sample_test.reset_index(inplace=True,drop=True)
        return sample_train, sample_test
    
    def get_datasets(self):
        return self.train_dataset, self.test_dataset 
        
    def get_visual_features(self,images):
        #preprocess image
        images, sizes, scales_yx = self.image_preprocess(images)
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
        return [normalized_boxes, features]
    
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
    
    def process_dataset(self,dataset):
        dataset['inputs'] = dataset['question'].apply(lambda x: self.get_text_features(x))
        boxes_features = dataset['image'].apply(lambda x: self.get_visual_features(x))
        dataset['normalized_boxes'] = boxes_features.apply(lambda x: x[0])
        dataset['features'] = boxes_features.apply(lambda x: x[1])
        return dataset
    
class MyTrainer():
    def __init__(self,model,processed_train,processed_test):
        self.model = model
        self.train = processed_train
        self.test = processed_test        
    
    def train_model(self,epochs=1):
        optim = AdamW(self.model.parameters(), lr=5e-5)
        train_loader = DataLoader(self.train, batch_size=2, shuffle=True)
        for epoch in range(1):
            for item in train_loader:
                optim.zero_grad()
                outputs = self.model.forward(item)
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
    
    def forward(self,item):
        #print(item)
        input_ids = item['input_ids']
        attention_mask=item['attention_mask']
        token_type_ids=item['token_type_ids']
        features = item['features']
        normalized_boxes = item['normalized_boxes']
        #print(inputs)
        print(input_ids.shape)
        print(attention_mask.shape)
        print(token_type_ids.shape)
        print(features.shape)
        print(normalized_boxes.shape)
        
        label = item['label']
        output = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            visual_feats=features,
            visual_pos=normalized_boxes,
            token_type_ids=token_type_ids,
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
        output = self.forward(item)
        m = torch.nn.Softmax(dim=1)
        probs = m(output.logits)
        print(probs)
        return output
        

#if __name__ == "__main__":
#if __name__ == "__main__":
task = 'train'
#task = 'test'
if task =='train':
    model = Lxmert()
    train, test = MyDataLoader().get_datasets()
    trainer = MyTrainer(model,train, test)
    trainer.train_model()
    model.save_model("my_model")
    output = model.run()
    
elif task =='test':
    model = Lxmert()
    model.load_model("my_model")
    output = model.run()
    