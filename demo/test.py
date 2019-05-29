from maskrcnn_benchmark.config import cfg
from predictor import COCODemo
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import numpy as np
import requests
import torch
import torchvision
import cv2

# config_file = "../configs/caffe2/e2e_mask_rcnn_R_50_FPN_1x_caffe2.yaml"
config_file = "../configs/rpn_X_101_32x8d_FPN_1x.yaml"

def load(url):
    """
    Given an url of an image, downloads the image and
    returns a PIL image
    """
    pil_image = Image.open(url).convert("RGB")
    # convert to BGR format
    image = np.array(pil_image)[:, :, [2, 1, 0]]
    return image

def draw(img, bbox):
    bbox_num = bbox.shape[0]
    for item in range(bbox_num):
        a = (bbox[item,0],bbox[item,1])
        b = (bbox[item,2],bbox[item,1])
        c = (bbox[item,2],bbox[item,3])
        d = (bbox[item,0],bbox[item,3])
        cv2.line(img,a,b,(0,255,0),3)
        cv2.line(img,b,c,(0,255,0),3)
        cv2.line(img,c,d,(0,255,0),3)
        cv2.line(img,d,a,(0,255,0),3)
    cv2.imwrite('2.png', img)
    

# update the config options with the config file
cfg.merge_from_file(config_file)
# manual override some options
cfg.merge_from_list(["MODEL.DEVICE", "cpu"])

coco_demo = COCODemo(
    cfg,
    min_image_size=800,
    confidence_threshold=0.7,
)
# load image and then run prediction
image = load('./1.jpg')
predictions = coco_demo.compute_prediction(image)
<<<<<<< HEAD
#plt.imshow(predictions[:,:,[2,1,0]])
#plt.axis('off')
#plt.savefig('2.png')
print(predictions.bbox.numpy())
=======
print(predictions.bbox.numpy())
draw(image,predictions)
# plt.imshow(predictions[:,:,[2,1,0]])
# plt.axis('off')
# plt.savefig('2.png')


>>>>>>> origin/master
