import torch
import pandas as pd
import numpy as np
from datasets import Dataset, load_dataset, Features
from sklearn.model_selection import train_test_split
from transformers import LxmertTokenizer, LxmertConfig, LxmertModel, LxmertForQuestionAnswering, PretrainedConfig
from modeling_frcnn import GeneralizedRCNN
import utils
from processing_image import Preprocess
from transformers import Trainer, TrainingArguments, AdamW
from torch.utils.data import DataLoader

class MyDataLoader():
    def __init__(self):
        self.lxmert_tokenizer = LxmertTokenizer.from_pretrained("unc-nlp/lxmert-base-uncased")
        self.rcnn_cfg = utils.Config.from_pretrained("unc-nlp/frcnn-vg-finetuned")
        self.rcnn = GeneralizedRCNN.from_pretrained("unc-nlp/frcnn-vg-finetuned", config=self.rcnn_cfg)
        self.image_preprocess = Preprocess(self.rcnn_cfg)
        self.train, self.test = self.read_datasets()
        self.processed_train = self.process_dataset(self.train)
        self.processed_test = self.process_dataset(self.test)
        return
    
    def read_datasets(self,data_path='./e-ViL/data/'):
        train = pd.read_csv(data_path+'esnlive_train.csv')
        labels_encoding = {'contradiction':torch.Tensor([1.,0.,0.]),'neutral': torch.Tensor([0.,1.,0.]),
                           'entailment':torch.Tensor([0.,0.,1.])}

        train = train[['hypothesis','Flickr30kID','gold_label']]
        train['gold_label']=train['gold_label'].apply(lambda label: labels_encoding[label])
        train['Flickr30kID'] = train['Flickr30kID'].apply(lambda x: data_path+'flickr30k_images/flickr30k_images/'+x)
        train.rename(columns={ train.columns[0]: "question", train.columns[1]: "image",
                              train.columns[2]: "label" }, inplace = True)
        #features = Features({k:v[0] for k,v in pd.DataFrame(train.dtypes).T.to_dict('list').items()})
        sample = train.sample(n=4, random_state=1)
        sample_train, sample_test = train_test_split(sample, test_size=0.2)
        sample_train.reset_index(inplace=True,drop=True)
        sample_test.reset_index(inplace=True,drop=True)
        #return Dataset.from_pandas(sample_train), Dataset.from_pandas(sample_test)
        #return Dataset.from_pandas(sample_train, features = features), Dataset.from_pandas(sample_test, features = features)
        return sample_train, sample_test
    
    def get_processed_datasets(self):
        return self.processed_test, self.processed_test
        
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
    
    def process_dataset(self,dataset):
        dataset['inputs'] = dataset['question'].apply(lambda x: self.get_text_features(x))
        boxes_features = dataset['image'].apply(lambda x: self.get_visual_features(x))
        dataset['normalized_boxes'] = boxes_features.apply(lambda x: x[0])
        dataset['features'] = boxes_features.apply(lambda x: x[1])
        return dataset
    
class MyDataset(torch.utils.data.Dataset):
    def __init__(self,inputs,features, normalized_boxes,labels):
        self.inputs = inputs
        self.features = features
        self.normalized_boxes = normalized_boxes
        self.labels = labels
    
    def __getitem__(self, idx):
        item = {'inputs':self.inputs[idx],'features':self.features[idx],
                'normalized_boxes': self.normalized_boxes[idx]}
        item['labels'] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)
    
    
class MyTrainer():
    def __init__(self,model,processed_train,processed_test):
        self.trainer = Trainer(model=model, train_dataset=self.train_dataset, 
                               eval_dataset=self.test_dataset)
    
    def train_model(self):
        #self.trainer.train()
        optim = AdamW(model.parameters(), lr=5e-5)
        train_loader = DataLoader(self.train_dataset, batch_size=16, shuffle=True)
        for epoch in range(1):
            print(epoch)
            k=0
            for batch in train_loader:
                print(k)
                k+=1
                print("zerograd")
                optim.zero_grad()
                #input_ids = batch['input_ids']
                #attention_mask = batch['attention_mask']
                questions = batch['questions']
                images = batch['images']
                labels = batch['labels']
                print("outputs")
                outputs = model.forward(questions,images,labels)
                #outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs[0]
                print("backward")
                loss.backward()
                print("optim")
                optim.step()
        model.eval()
        model.save_pretrained("my_model")
        return 
        
    
class Lxmert(LxmertModel):
    def __init__(self,config,numb_labels=3):
        super().__init__(LxmertConfig.from_pretrained(config))
        #self.lxmert = LxmertModel.from_pretrained("unc-nlp/lxmert-base-uncased")
        self.new_encoder_layer = torch.nn.TransformerEncoderLayer(d_model=768, nhead=8)
        self.new_transformer_encoder = torch.nn.TransformerEncoder(self.new_encoder_layer, num_layers=3)
        #
        self.config.problem_type = "multi_label_classification"
        self.classification = torch.nn.Linear(self.config.hidden_size, numb_labels)
        self.num_labels = numb_labels
        # don't forget to init the weights for the new layers
        #self.init_weights()
    
    def forward(self,inputs, features, normalized_boxes, labels):
        output = super().forward(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            visual_feats=features,
            visual_pos=normalized_boxes,
            token_type_ids=inputs['token_type_ids'],
            return_dict=True,
            output_attentions=False,
        )
                
        aux = self.classification(output.pooled_output[0])
        output.logits = aux
        output.loss = None
        
        if self.config.problem_type == "multi_label_classification":
          loss_fct = torch.nn.BCEWithLogitsLoss()
          output.loss = loss_fct(output.logits, labels)
        elif self.config.problem_type == "regression":
          loss_fct = torch.nn.MSELoss()
          if self.num_labels == 1: output.loss = loss_fct(output.logits.squeeze(), labels.squeeze())
          else: output.loss = loss_fct(output.logits, labels)
        elif self.config.problem_type == "single_label_classification":
          loss_fct = torch.nn.CrossEntropyLoss()
          output.loss = loss_fct(output.logits.view(-1, self.num_labels), labels.view(-1)) 
        return output
        
        
    def run(self,dataset):
        img_path = dataset.loc[5,'image']#"32542645.jpg"
        question = dataset.loc[5,'question']#"How many people are in the image?"
        label = dataset.loc[5,'label']
        #print(self.forward(question,img_path))
        #self.train(sample_train,sample_test)
        return self.forward(question,img_path,label)
        

#if __name__ == "__main__":
model = Lxmert("unc-nlp/lxmert-base-uncased")
train, test = MyDataLoader().get_processed_datasets()
train = MyDataset(train['inputs'].values,train['features'].values,train['normalized_boxes'].values,
                  train['label'].values)
test = MyDataset(test['inputs'].values,test['features'].values,test['normalized_boxes'].values,
                  test['label'].values)
#trainer = MyTrainer(model,train, test)

train_loader = DataLoader(train, batch_size=16, shuffle=True)
for epoch in range(1):
    for batch in train_loader:
        inputs = batch['inputs']
        features= batch['features']
        normalized_boxes = batch['normalized_boxes']
        labels = batch['labels']
        outputs = model.forward(inputs, features, normalized_boxes, labels)
        break

#trainer.train_model()
"""
output = model.run(train)
print(output)
"""
