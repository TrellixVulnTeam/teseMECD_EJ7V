import torch
import utils
from modeling_frcnn import GeneralizedRCNN
from processing_image import Preprocess

def norm_box_new(boxes, raw_sizes):
    if not isinstance(boxes, torch.Tensor):
        normalized_boxes = boxes.copy()
    else:
        normalized_boxes = boxes.clone()
    print(normalized_boxes[:, :, (0, 2)].shape)
    print(raw_sizes[:, 1].shape)
    normalized_boxes[:, :, (0, 2)] /= raw_sizes[:, 1].view(-1,1,1)
    normalized_boxes[:, :, (1, 3)] /= raw_sizes[:, 0].view(-1,1,1)
    return normalized_boxes

def norm_box(boxes, raw_sizes):
    if not isinstance(boxes, torch.Tensor):
        normalized_boxes = boxes.copy()
    else:
        normalized_boxes = boxes.clone()
    print(normalized_boxes[:, :, (0, 2)].shape)
    print(raw_sizes[:, 1].shape)
    normalized_boxes[:, :, (0, 2)] /= raw_sizes[:, 1]
    normalized_boxes[:, :, (1, 3)] /= raw_sizes[:, 0]
    return normalized_boxes

normalized_boxes = torch.ones(8,36,4)
raw_sizes = torch.ones(8,2)
norm_box(normalized_boxes,raw_sizes)
norm_box_new(normalized_boxes,raw_sizes)
#norm_box_new(torch.ones(4,36,4),torch.ones(4,2))
#norm_box(torch.ones(4,36,4),torch.ones(4,2))
#norm_box(torch.ones(1,36,4),torch.ones(1,2))
#norm_box(torch.ones(2,36,4),torch.ones(1,2))

"""
#Size 4
img = ["../e-ViL/data/flickr30k_images/flickr30k_images/36979.jpg",
          "../e-ViL/data/flickr30k_images/flickr30k_images/65567.jpg",
          "../e-ViL/data/flickr30k_images/flickr30k_images/81641.jpg",
          "../e-ViL/data/flickr30k_images/flickr30k_images/134206.jpg"]
"""

img = ["../e-ViL/data/flickr30k_images/flickr30k_images/36979.jpg"]

rcnn_cfg = utils.Config.from_pretrained("unc-nlp/frcnn-vg-finetuned")
rcnn = GeneralizedRCNN.from_pretrained("unc-nlp/frcnn-vg-finetuned", config=rcnn_cfg)
image_preprocess = Preprocess(rcnn_cfg)

images, sizes, scales_yx = image_preprocess(img)

print("IMAGES ",images.shape)
print("Sizes ", sizes)
print("Scales_yx ",scales_yx)

#preprocess image
output_dict = rcnn(
    images, 
    sizes, 
    scales_yx=scales_yx, 
    padding="max_detections",
    max_detections=rcnn_cfg.max_detections,
    return_tensors="pt"
)

text = ["Hello","Hello2","Hello3","Hello4"]
label = torch.Tensor([0,1,2,0])
