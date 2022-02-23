# Computes the embeddings of the images in the test split of InstaCities1M
# Saved them in a txt where each line contains an image ID and its empedding

import model
import dataset
import torch 
import numpy as np

dataset_root = '../../../datasets/InstaCities1M/'
model_path = dataset_root + 'models/InstaCities1M_best.pth.tar'
split_test = 'test_InstaCities1M.txt'
embedding_dimensionality = 400
batch_size = 256 # 64
num_workers = 8
gpu = 0
output_file_path = dataset_root + 'embeddings/test_img_embeddings.txt'
output_file = open(output_file_path, 'w')

model_test = model.Model_Test(embedding_dimensionality).cuda(gpu)
model_test = torch.nn.DataParallel(model_test, device_ids=[gpu]).cuda(gpu)
state_dict = torch.load(model_path)
model_test.load_state_dict(state_dict, strict=True)

test_dataset = dataset.Dataset(dataset_root, split_test, embedding_dimensionality)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

with torch.no_grad():
    model_test.eval()
    for i, (img_name, img, target) in enumerate(test_loader):
        img = torch.autograd.Variable(img)
        outputs = model_test(img)
        for batch_idx, img_embedding in enumerate(outputs):
            out_string = img_name[batch_idx]
            img_embedding = np.array(img_embedding.cpu())
            for x in range(0,embedding_dimensionality): 
            	out_string += ',' + str(img_embedding[x])
            output_file.write(out_string + '\n')

        print(str(i) + ' / ' + str(len(test_loader)))

output_file.close()

print("Done")