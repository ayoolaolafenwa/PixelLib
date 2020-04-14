import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
from .mask_rcnn  import MaskRCNN
from .coco import CocoConfig as inferconfig
import pixellib.visualize


class InferenceConfig(inferconfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU

    
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    

config = InferenceConfig()


class instance_segmentation():

    def __init__(self):
        self.MODEL_DIR = os.getcwd()
        
    def load_model(self, model_path):
        self.model = MaskRCNN(mode = "inference", model_dir = self.MODEL_DIR, config = config)
        self.model.load_weights(model_path, by_name= True)
    
    def segmentImage(self, image_path, output_image_name, show_boxes = True):
        if show_boxes == True:
            image = skimage.io.imread(image_path)
            # Run detection
            results = self.model.detect([image], verbose=1)

            class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']
               
        
            r = results[0]
            output = pixellib.visualize.display_instances(image, output_image_name, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])
            return output
            

        elif show_boxes == False:
            image = skimage.io.imread(image_path)
            # Run detection
            results = self.model.detect([image], verbose=1)
               
        
            r = results[0]
            output = pixellib.visualize.display_instances2(image, output_image_name, r['rois'], r['masks'], r['class_ids'])
            

            return output
            
 
        

