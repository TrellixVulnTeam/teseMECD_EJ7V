import requests
import torch
from transformers import CLIPProcessor, CLIPTokenizer, CLIPModel, CLIPTextModel, CLIPVisionModel, CLIPConfig
from transformers import VisionEncoderDecoderConfig, VisionEncoderDecoderModel
from transformers import VisionTextDualEncoderModel, VisionTextDualEncoderProcessor
from PIL import Image
from my_trainer import MyTrainer

processor = CLIPProcessor.from_pretrained("flax-community/clip-rsicd-v2")
tokenizer = CLIPTokenizer.from_pretrained("flax-community/clip-rsicd-v2")

class MyVisionTextModel(CLIPModel):
    def __init__(self, config, num_labels=10):
      super().__init__(CLIPConfig.from_pretrained("flax-community/clip-rsicd-v2"))
      self.config.problem_type = "single_label_classification"
      self.new_encoder_layer = torch.nn.TransformerEncoderLayer(d_model=512, nhead=8)
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
    
    def forward(self,input_ids,pixel_values,attention_mask,position_ids,return_loss,output_attentions,output_hidden_states,label):
      output = super().forward(input_ids,  pixel_values, attention_mask, position_ids, return_loss, output_attentions, output_hidden_states, return_dict=True)
      aux_vision = output.vision_model_output[0]
      aux_vision = self.visual_projection(aux_vision)
      aux_text = output.text_model_output[0]
      aux_text = self.text_projection(aux_text)
      aux = torch.cat((aux_vision,aux_text),dim=1)
      aux_mask=attention_mask
      aux_mask = torch.cat(( torch.ones(aux_vision.size,dtype=torch.bool) , aux_mask), dim=1 )	
      aux = self.new_transformer_encoder( aux, mask=aux_mask )
      input_mask_expanded = aux_mask.unsqueeze(-1).expand(aux.size()).float()
      aux = torch.sum(aux * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
      aux = self.classification(aux)
      output.logits = aux
      output.loss = None
      output.loss = self.output_loss(output, label)
      return output
  
    def save_model(self,path):
        torch.save(self.state_dict(), path)
        
    def load_model(self,path):
        self.load_state_dict(torch.load(path))
        self.eval()

def run(model):
    pass

#if __name__ == "__main__":
task = 'train'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#task = 'test'

if task =='train':
    model = MyVisionTextModel()
    model = model.to(device)
    train = MyDataset("my_train_db")
    test = MyDataset("my_test_db")
    trainer = MyTrainer(model,train, test, device = device)
    trainer.train_model()
    model.save_model("my_model2")
    run(model)
elif task =='test':
    model = MyVisionTextModel()
    model = model.to(device)
    model.load_model("my_model2")
    run(model)
    

#model.text_model = clip_model.text_model
#model.vision_model = clip_model.vision_model
#model.visual_projection = clip_model.visual_projection
#model.text_projection = clip_model.text_projection
#model.logit_scale = clip_model.logit_scale

url = "https://fki.tic.heia-fr.ch/static/img/a01-122-02.jpg"
image = Image.open(requests.get(url, stream=True).raw).convert("RGB")

# clip training example
pixel_values = processor(images=image, return_tensors="pt").pixel_values
text = "hello world"
inputs = processor.tokenizer(text, return_tensors="pt")
outputs = model(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask, pixel_values=pixel_values, return_loss=True )

loss = outputs.loss

