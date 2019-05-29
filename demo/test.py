from maskrcnn_benchmark.config import cfg
from predictor import COCODemo
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import numpy as np
import requests

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
#plt.imshow(predictions[:,:,[2,1,0]])
#plt.axis('off')
#plt.savefig('2.png')
print(predictions.bbox)
