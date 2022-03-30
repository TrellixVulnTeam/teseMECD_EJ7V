import pandas as pd

#-----------------
#---Load Data-----
#-----------------
data_path = './e-ViL/data/'
train = pd.read_csv(data_path+'esnlive_train.csv')
test = pd.read_csv(data_path+'esnlive_test.csv')
results = pd.read_csv(data_path+'flickr30k_images/results.csv', sep ='|')
#train_json = pd.read_json(data_path+'/esnlive/esnlive_train.json')

#------------------
#------EDA---------
#------------------ 
"""
from PIL import Image
  
im_path = data_path+'flickr30k_images/flickr30k_images/'
im_name = train.loc[90,'Flickr30kID']
im = Image.open(im_path+im_name) 
anottations = results.loc[results['image_name'] == im_name]
print(anottations)
# This method will show image in any image viewer 
im.show()
"""
#-------------------
#-----Embeddings----
#-------------------

#-------------------
#-----ViT ---
#-------------------
"""
from transformers import ViTModel, ViTConfig

# Initializing a ViT vit-base-patch16-224 style configuration
configuration = ViTConfig()

# Initializing a model from the vit-base-patch16-224 style configuration
model = ViTModel(configuration)
"""
