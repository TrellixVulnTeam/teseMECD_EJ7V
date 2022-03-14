import torch 
from transformers import AdamW
from torch.utils.data import DataLoader

class MyTrainer():
    def __init__(self,model,train,test, device =None):
        if device == None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device#'cpu'
        self.model = model
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