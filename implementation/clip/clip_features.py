import requests
import torch
from transformers import CLIPProcessor, CLIPTokenizer, CLIPModel, CLIPTextModel, CLIPVisionModel, CLIPConfig
from transformers import VisionEncoderDecoderConfig, VisionEncoderDecoderModel
from transformers import VisionTextDualEncoderModel, VisionTextDualEncoderProcessor
from PIL import Image
import pickle
import lmdb
import pandas as pd

from transformers import AdamW
from torch.utils.data import DataLoader

class MyTrainer():
    def __init__(self,model,train,test, device =None):
        if device == None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        self.model = model
        self.optimizer = AdamW(self.model.parameters(), lr=5e-5)
        self.train = train
        self.test = test     
    
    def train_model(self,epochs=1):
        train_loader = DataLoader(self.train, batch_size=1, shuffle=True)
        for epoch in range(epochs):
            for item in train_loader:
                input_ids = item['input_ids'].to(self.device)
                attention_mask=item['attention_mask'].to(self.device)
                token_type_ids=item['token_type_ids'].to(self.device)
                features = item['features'].to(self.device)
                normalized_boxes = item['normalized_boxes'].to(self.device)
                label = item['label'].to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model.forward(input_ids,attention_mask,token_type_ids,
                                             features,normalized_boxes,label)
                loss = outputs.loss#[0]
                loss.backward()
                self.optimizer.step()
        self.model.eval()
        self.model.save_pretrained("my_model")
        return
    
class MyDataset(torch.utils.data.Dataset):
    def __init__(self,path):
        self.path = path
        self.dataset = pd.read_csv(self.path)
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
            
    
    def __getitem__(self, idx):
        item = self.dataset
        item = self.dataset
        return item

    def __len__(self):
        return self.size
    
class MyVisionTextModel(CLIPModel):
    def __init__(self, num_labels=3):
      super().__init__(CLIPConfig.from_pretrained("openai/clip-vit-base-patch32"))
      self.config.problem_type = "single_label_classification"
      self.new_encoder_layer = torch.nn.TransformerEncoderLayer(d_model=512, nhead=8, dropout=0.1)
      self.new_transformer_encoder = torch.nn.TransformerEncoder(self.new_encoder_layer, num_layers=3)
      self.classification = torch.nn.Linear(512, num_labels, bias=True)
      self.num_labels = num_labels
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
    
    def get_visual_features():
        return
    
    def get_text_features():
        return
    
    def forward(self, input_ids=None, pixel_values=None, attention_mask=None, position_ids=None, return_loss=None, output_attentions=None, output_hidden_states=None, label=None, ):
      output = super().forward(input_ids,  pixel_values, attention_mask, position_ids, return_loss, output_attentions, output_hidden_states, return_dict=True)
      print('ov',output.vision_model_output[0].size())
      print('ot',output.text_model_output[0].size())
      print('am',attention_mask.size())
      aux_vision = output.vision_model_output[0]#.pooler_output#
      aux_vision = self.visual_projection(aux_vision)
      aux_text = output.text_model_output[0]#.pooler_output#[0]
      aux_text = self.text_projection(aux_text)
      aux = torch.cat((aux_vision,aux_text),dim=1)

      ones = torch.ones(1,aux_vision.shape[1],dtype=torch.float)
      aux_mask = torch.cat((ones,attention_mask), dim=1)
      padding_mask = torch.swapaxes(aux_mask, 0, 1)

      print('aux',aux.size())
      print('aux_mask',aux_mask.size())

      aux = self.new_transformer_encoder( aux, src_key_padding_mask= padding_mask)
      #aux = self.new_transformer_encoder( aux, mask= padding_mask)
      
      input_mask_expanded = aux_mask.unsqueeze(-1).expand(aux.size()).float()
      aux = torch.sum(aux * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
      
      aux = self.classification(aux)
      output.logits = aux
      output.loss = self.output_loss(output, label)
      return output
  
    def save_model(self,path):
        torch.save(self.state_dict(), path)
        
    def load_model(self,path):
        self.load_state_dict(torch.load(path))
        self.eval()

def run(model):
    processor = CLIPProcessor.from_pretrained("flax-community/clip-rsicd-v2",local_files_only = True)
    url = "test.jpg"
    image = Image.open(url).convert("RGB")
    # clip training example
    pixel_values = processor(images=image, return_tensors="pt").pixel_values
    text = "hello world"
    inputs = processor.tokenizer(text, return_tensors="pt")
    outputs = model(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask, 
                    pixel_values=pixel_values, return_loss=True,
                    label = torch.LongTensor([2]))
    loss = outputs.loss
    logits = outputs.logits
    print(loss)
    print(logits)
    return outputs

#task = 'train'
task = 'test'
device = 'cpu'#torch.device("cuda" if torch.cuda.is_available() else "cpu")

#model.text_model = clip_model.text_model
#model.vision_model = clip_model.vision_model
#model.visual_projection = clip_model.visual_projection
#model.text_projection = clip_model.text_projection
#model.logit_scale = clip_model.logit_scale

if task =='train':
    model = MyVisionTextModel()
    model = model.to(device)
    train = MyDataset("../my_train_db")
    test = MyDataset("../my_test_db")
    trainer = MyTrainer(model,train, test, device = device)
    trainer.train_model()
    model.save_model("my_model2")
elif task =='test':
    model = MyVisionTextModel()
    model = model.to(device)
    url = "test.jpg"
    image = Image.open(url).convert("RGB")
    #preprocessing
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32",local_files_only = True)
    pixel_values = processor(images=image, return_tensors="pt").pixel_values
    text = "hello world"
    inputs = processor.tokenizer(text, return_tensors="pt")
    #forward-pass
    outputs = model(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask, 
                    pixel_values=pixel_values, return_loss=True,
                    label = torch.LongTensor([2]))
    

