import torch 
from transformers import AdamW
from torch.utils.data import DataLoader
import deepspeed
import argparse

def add_argument():
    parser=argparse.ArgumentParser(description='MyTrainer')
    #data
    # cuda
    #parser.add_argument('--with_cuda', default=True, action='store_true',
    #                    help='use CPU in case there\'s no GPU support')
    # train
    parser.add_argument('-b', '--batch_size', default=1, type=int,
                        help='mini-batch size (default: 32)')
    parser.add_argument('-e', '--epochs', default=1, type=int,
                        help='number of total epochs (default: 30)')
    parser.add_argument('-g', '--num_gpus', default=1, type=int,
                        help='number of total gpus (default: 1)')
    # Include DeepSpeed configuration arguments
    parser = deepspeed.add_config_arguments(parser)
    
    args=parser.parse_args()

    return args
 
class MyTrainer():
    def __init__(self,model,train, cmd_args = add_argument()):
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
            input_ids = item['input_ids'].to(self.model_engine.device)
            attention_mask=item['attention_mask'].to(self.model_engine.device)
            token_type_ids=item['token_type_ids'].to(self.model_engine.device)
            features = item['features'].to(self.model_engine.device)
            normalized_boxes = item['normalized_boxes'].to(self.model_engine.device)
            label = item['label'].to(self.model_engine.device)
            
            #self.optimizer.zero_grad()
            #outputs = self.model.forward(input_ids,attention_mask,token_type_ids,
            #                             features,normalized_boxes,label)
            #loss = outputs.loss#[0]
            #loss.backward()
            #self.optimizer.step()
            
            outputs = self.model_engine(input_ids,attention_mask,token_type_ids,
                                         features,normalized_boxes,label)
            loss = outputs.loss
            
            self.model_engine.backward(loss)
            self.model_engine.step()
            
        self.model.eval()
        self.model.save_pretrained("my_model")
        return