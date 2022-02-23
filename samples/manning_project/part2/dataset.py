# Data loader. Loads all the InstaCities1M image IDs and caption embeddings and then provides them during training or testing

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms


class Dataset(Dataset):

    def __init__(self, root_dir, split, embedding_dimensionality):

        self.root_dir = root_dir
        self.split = split
        self.embedding_dimensionality = embedding_dimensionality

        self.preprocess = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        #transforms.Resize(256), # If we want the crop to be more centered
        #transforms.CenterCrop(224), # This makes more sense for testing
        transforms.RandomCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # ImageNet stats
        ])

        print("Loading data from " + str(split))
        gt_file = root_dir + 'embeddings/' + split
        # Count number of images in split
        num_lines = 0
        with open(gt_file, 'r') as f:
            for i, l in enumerate(f):
                pass
        num_lines = i + 1

        # Do small test
        # num_lines = 640 * 4 # Train only with x images to test code 
        # if 'val' in self.split:
        #     num_lines = 128

        # Load img IDs and caption embeddings to memory
        print("Num lines: " + str(num_lines))
        self.img_ids = np.empty([num_lines], dtype="S50")
        self.captions_embeddings = np.zeros((num_lines, self.embedding_dimensionality), dtype=np.float32)

        print("Loading labels ...")
        with open(gt_file, 'r') as annsfile:
            for c, i in enumerate(annsfile):
                if c == num_lines: break
                data = i.split(',')
                self.img_ids[c] = data[0]
                # Load caption word2vec embedding
                for l in range(0, self.embedding_dimensionality):
                    self.captions_embeddings[c, l] = float(data[l + 1])

        print("Data read.")


    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):

        ### Image
        img_name = self.img_ids[idx].decode('utf-8')
        input_img = Image.open(self.root_dir + 'img_resized_1M/cities_instagram/' + img_name + '.jpg').convert('RGB')
        img_tensor = self.preprocess(input_img) # Data augmentaion + preprocessing

        ### Target Vector
        target_tensor = torch.from_numpy(self.captions_embeddings[idx,:])

        return img_name, img_tensor, target_tensor