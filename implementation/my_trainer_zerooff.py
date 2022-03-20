import torch 
from transformers import AdamW
from torch.utils.data import DataLoader
import deepspeed

class MyTrainer():
    def __init__(self,model,train, cmd_args):
        self.train = train
        self.model = model
        self.optimizer = AdamW
        self.model_engine, self.model_optimizer, self.training_dataloader, _ = deepspeed.initialize(args=cmd_args,
                                                     model=self.model,
                                                     model_parameters=self.model.parameters(),
                                                     optimizer = self.optimizer)
        
    
    def train_model(self):
        for i, item in enumerate(self.training_dataloader):
         # get the inputs; data is a list of [inputs, labels]
         inputs = data[0].to(model_engine.device)
         labels = data[1].to(model_engine.device)
        
         outputs = model_engine(inputs)
         loss = criterion(outputs.loss, labels)
        
         model_engine.backward(loss)
         model_engine.step()

                input_ids = item['input_ids'].to(self.model_engine.device)
                attention_mask=item['attention_mask'].to(self.model_engine.device)
                token_type_ids=item['token_type_ids'].to(self.model_engine.device)
                features = item['features'].to(self.model_engine.device)
                normalized_boxes = item['normalized_boxes'].to(self.model_engine.device)
                label = item['label'].to(self.model_engine.device)
                self.optimizer.zero_grad()
                outputs = self.model.forward(input_ids,attention_mask,token_type_ids,
                                             features,normalized_boxes,label)
                loss = outputs.loss#[0]
                loss.backward()
                self.optimizer.step()
        self.model.eval()
        self.model.save_pretrained("my_model")
        return