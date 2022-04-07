import torch
import utils
from transformers import LxmertTokenizer
from modeling_frcnn import GeneralizedRCNN
from processing_image import Preprocess
import pandas as pd
import lmdb
import io
import pickle

class DataWriter():
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lxmert_tokenizer = LxmertTokenizer.from_pretrained("unc-nlp/lxmert-base-uncased")
        self.rcnn_cfg = utils.Config.from_pretrained("unc-nlp/frcnn-vg-finetuned")
        self.rcnn_cfg.MODEL.DEVICE = self.device
        self.rcnn = GeneralizedRCNN.from_pretrained("unc-nlp/frcnn-vg-finetuned", config=self.rcnn_cfg)
        self.image_preprocess = Preprocess(self.rcnn_cfg)
        self.train_dataset = self.read_dataset('esnlive_train.csv')
        self.test_dataset = self.read_dataset('esnlive_test.csv')
        self.dev_dataset = self.read_dataset('esnlive_dev.csv')
        return
    
    def read_dataset(self, dataset_name ,data_path = '../e-ViL/data/',
                     images_path ='flickr30k_images/flickr30k_images/' ):
        dataset = pd.read_csv(data_path+dataset_name)
        labels_encoding = {'contradiction':0,'neutral': 1,
                           'entailment':2}
        dataset = dataset[['hypothesis','Flickr30kID','gold_label']]
        dataset['gold_label']=dataset['gold_label'].apply(lambda label: labels_encoding[label])
        dataset['Flickr30kID'] = dataset['Flickr30kID'].apply(lambda x: data_path+images_path+x)
        return dataset
        
    def get_visual_features(self,img):
        #preprocess image
        images, sizes, scales_yx = self.image_preprocess(img)
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
            max_length=190,
            truncation=True,
            return_token_type_ids=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt"
        )
        return inputs
    
    def get_tensor_as_bytes(self,tensor,buffer):
        torch.save(tensor, buffer)
        return buffer.getvalue()
        
    def write_to_lmdb(self,pd_dataframe,filename,map_size = 1000000000):#1GB
        k = 0
        env = lmdb.open(filename, map_size= map_size)
        txn = env.begin(write=True)
        buf = io.BytesIO()
        for idx in range(len(pd_dataframe)):
            if(k==10):
                break
            print(k)
            k+=1
            img = pd_dataframe.loc[idx,'Flickr30kID']
            text = pd_dataframe.loc[idx,'hypothesis']
            label = pd_dataframe.loc[idx,'gold_label']
            inputs = self.get_text_features(text)
            
            normalized_boxes, features = self.get_visual_features(img)
            item = {'input_ids': inputs['input_ids'].numpy(),
                    'attention_mask': inputs['attention_mask'].numpy(),
                    'token_type_ids': inputs['token_type_ids'].numpy(),
                    'features': features.numpy(),
                    'normalized_boxes': normalized_boxes.numpy(),
                    'label': label}
            txn.put(key = str(idx).encode(), value = pickle.dumps(item))
        # Commit changes through the commit() function 
        txn.commit() 
        env.close()


def deserializeItem(item):
    item = pickle.loads(item)
    item['input_ids']=torch.IntTensor(item['input_ids'][0])
    item['attention_mask']=torch.IntTensor(item['attention_mask'][0])
    item['token_type_ids']=torch.IntTensor(item['token_type_ids'][0])
    item['normalized_boxes']=torch.FloatTensor(item['normalized_boxes'][0])
    item['features']=torch.FloatTensor(item['features'][0])
    return item

def ReadItem():
    env = lmdb.open("my_train_db")
    txn = env.begin()
    item = txn.get(str(2).encode())
    item = deserializeItem(item)
    env.close()
    return item
    
def ReadAllData():
    import sys
    env = lmdb.open("my_train_db") 
    txn = env.begin()
    total_bytes = 0
    # Traverse all data and key values through cursor() 
    for key, value in txn.cursor():
        total_bytes+=sys.getsizeof(value)
        total_bytes+=sys.getsizeof(key)
    print(total_bytes)
    env.close()

def getStats(lmdb_file_name="my_train_db"):
    lmdb_env = lmdb.open(lmdb_file_name, readonly=True)
    stats = lmdb_env.stat()
    info = lmdb_env.info()
    lmdb_env.close()
    return stats, info

def getSize(lmdb_file_name="my_train_db"):
    lmdb_env = lmdb.open(lmdb_file_name, readonly=True)
    stats = lmdb_env.stat()
    info = lmdb_env.info()
    lmdb_env.close()
    dbSize = stats['psize'] * (stats['leaf_pages'] + stats['branch_pages'] + stats['overflow_pages'])
    return dbSize/1024/1024
"""
dw = DataWriter()
dw.write_to_lmdb(dw.train_dataset, 'my_train_db',map_size = 1073741824)#1GB
dw.write_to_lmdb(dw.train_dataset, 'my_train_db',map_size = 4194304)#4MB
dw.write_to_lmdb(dw.train_dataset, 'my_train_db',map_size = 9252000000)#9.252GB
dw.write_to_lmdb(dw.test_dataset, 'my_test_db',map_size = 1000000000)
"""

   