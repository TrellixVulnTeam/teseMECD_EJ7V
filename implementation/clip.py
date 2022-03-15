import requests
import torch
from transformers import CLIPProcessor, CLIPTokenizer, CLIPModel, CLIPTextModel, CLIPVisionModel, CLIPConfig
from transformers import VisionEncoderDecoderConfig, VisionEncoderDecoderModel
from transformers import VisionTextDualEncoderModel, VisionTextDualEncoderProcessor
from PIL import Image
from my_trainer import MyTrainer

"""
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
"""
    
class MyVisionTextModel(CLIPModel):
    def __init__(self, num_labels=10):
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
    
    def forward(self, input_ids=None, pixel_values=None, attention_mask=None, position_ids=None, return_loss=None, output_attentions=None, output_hidden_states=None, label=None, ):
      output = super().forward(input_ids,  pixel_values, attention_mask, position_ids, return_loss, output_attentions, output_hidden_states, return_dict=True)
      #print(output.vision_model_output)
      #print(output.vision_model_output[0])
      aux_vision = output.vision_model_output[0]#.pooler_output#
      aux_vision = self.visual_projection(aux_vision)
      print('attention_mask',attention_mask)
      print('aux_vision',aux_vision.size())
      aux_text = output.text_model_output[0]#.pooler_output#[0]
      print('aux_text',aux_text.size())
      aux_text = self.text_projection(aux_text)
      aux = torch.cat((aux_vision,aux_text),dim=1)
      print('cat_vision_text',aux.size())
      aux_mask=attention_mask
      print('attention_mask',aux_mask.size())
      ones = torch.ones(1,aux_vision.shape[1],dtype=torch.bool)
      print('ones',ones.size())
      print('here',ones.size(),aux_mask.size())
      print('TYPE1',ones.type(),'TYPE2',aux_mask.type())
      aux_mask = torch.cat((ones,aux_mask.bool()), dim=1)
      print('aux_mask',aux_mask.size())
      print('Aux_size',aux.size())
      aux = self.new_transformer_encoder( aux, mask=aux_mask )
      print('transformer_encoder',aux.size())
      input_mask_expanded = aux_mask.unsqueeze(-1).expand(aux.size()).float()
      aux = torch.sum(aux * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
      aux = self.classification(aux)
      output.logits = aux
      output.loss = None
      #output.loss = self.output_loss(output, label)
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
    outputs = model(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask, pixel_values=pixel_values, return_loss=True )
    loss = outputs.loss
    logits = outputs.logits
    print(loss)
    print(logits)
    return outputs

#if __name__ == "__main__":
#task = 'train'
task = 'test'
device = 'cpu'#torch.device("cuda" if torch.cuda.is_available() else "cpu")
#task = 'test'

if task =='train':
    model = MyVisionTextModel()
    model = model.to(device)
    #train, test = MyDataLoader().get_datasets()
    train = None
    test = None
    trainer = MyTrainer(model,train, test, device = device)
    trainer.train_model()
    model.save_model("my_model2")
    output = run(model)
elif task =='test':
    model = MyVisionTextModel()
    model = model.to(device)
    #model.load_model("my_model2")
    output = run(model)
    
#model.text_model = clip_model.text_model
#model.vision_model = clip_model.vision_model
#model.visual_projection = clip_model.visual_projection
#model.text_projection = clip_model.text_projection
#model.logit_scale = clip_model.logit_scale

