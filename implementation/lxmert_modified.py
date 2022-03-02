import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import LxmertTokenizer, LxmertConfig, LxmertModel, LxmertForQuestionAnswering, PretrainedConfig
from modeling_frcnn import GeneralizedRCNN
import utils
from processing_image import Preprocess
from transformers import Trainer
    
class Lxmert(LxmertModel):
    def __init__(self):
        super().__init__(LxmertConfig.from_pretrained("unc-nlp/lxmert-base-uncased"))
        self.config.problem_type = "single_label_classification"
        numb_labels = 3
        self.num_labels = numb_labels
        self.lxmert_tokenizer = LxmertTokenizer.from_pretrained("unc-nlp/lxmert-base-uncased")
        self.rcnn_cfg = utils.Config.from_pretrained("unc-nlp/frcnn-vg-finetuned")
        self.rcnn = GeneralizedRCNN.from_pretrained("unc-nlp/frcnn-vg-finetuned", config=self.rcnn_cfg)
        self.image_preprocess = Preprocess(self.rcnn_cfg)
        self.lxmert = LxmertModel.from_pretrained("unc-nlp/lxmert-base-uncased")
        self.classification = torch.nn.Linear(self.lxmert.config.hidden_size, numb_labels)
        # don't forget to init the weights for the new layers
        self.init_weights()
    
    def forward(self,text,img,labels):
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
        
        output = self.lxmert(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            visual_feats=features,
            visual_pos=normalized_boxes,
            token_type_ids=inputs.token_type_ids,
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
    
    def train(self,train,test):
        trainer = Trainer(model=self, train_dataset = train, eval_dataset = test)
        #, args=training_args
        trainer.train()
    
    def run(self):
        data_path = './e-ViL/data/'
        train = pd.read_csv(data_path+'esnlive_train.csv')
        labels_encoding = {'contradiction':torch.Tensor([1.,0.,0.]),'neutral': torch.Tensor([0.,1.,0.]),
                           'entailment':torch.Tensor([0.,0.,1.])}
        train['gold_label']=train['gold_label'].apply(lambda label: labels_encoding[label])
        #test = pd.read_csv(data_path+'esnlive_test.csv')
        #results = pd.read_csv(data_path+'flickr30k_images/results.csv', sep ='|')
        sample = train.sample(n=100, random_state=1)
        sample_train, sample_test = train_test_split(sample, test_size=0.2)
        sample_train.reset_index(inplace=True,drop=True)
        sample_test.reset_index(inplace=True,drop=True)
        #img_path = data_path+'flickr30k_images/flickr30k_images/'+"32542645.jpg"#train.loc[90,'Flickr30kID']
        #question = "How many people are in the image?"
        img_path = data_path+'flickr30k_images/flickr30k_images/'+ sample_train.loc[50,'Flickr30kID']#"32542645.jpg"
        question = sample_train.loc[50,'hypothesis'] #"How many people are in the image?"
        label = sample_train.loc[50,'gold_label']
        #print(self.forward(question,img_path))
        #self.train(sample_train,sample_test)
        return self.forward(question,img_path,label)
        

#if __name__ == "__main__":
model = Lxmert()
output = model.run()
print(output.logits)
print(output.loss)
train = pd.read_csv('./e-ViL/data/'+'esnlive_train.csv')
labels_encoding = {'contradiction':torch.Tensor([1.,0.,0.]),'neutral': torch.Tensor([0.,1.,0.]),
                   'entailment':torch.Tensor([0.,0.,1.])}
train['gold_label']=train['gold_label'].apply(lambda label: labels_encoding[label])
#from datasets import Dataset, load_dataset, Features
#features = Features({k:v[0] for k,v in pd.DataFrame(train.dtypes).T.to_dict('list').items()})
#train = Dataset.from_pandas(train)
"""
data_path = './e-ViL/data/'
data_files = {
    "train": data_path+'esnlive_train.csv',
    "validation": data_path+'esnlive_dev.csv',
    "test": data_path+'esnlive_test.csv'
}
dataset = load_dataset("csv", data_files=data_files)
train=dataset["train"]
test = dataset["test"]
"""
#train = load_dataset('csv', data_files=data_path+'esnlive_train.csv',split='train')
#test = load_dataset('csv', data_files=data_path+'esnlive_test.csv',split='train')
test = train
model.train(train,test)
print(output)