import cv2
import numpy as np
import random
import os
import sys
import math
from .mask_rcnn import MaskRCNN
from .coco import CocoConfig as inferconfig
import colorsys



class InferenceConfig(inferconfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


config = InferenceConfig()


class instance_segmentation():
    def __init__(self):
        self.model_dir = os.getcwd()
    def load_model(self, model_path):
        self.model = MaskRCNN(mode = "inference", model_dir = self.model_dir, config = config)
        self.model.load_weights(model_path, by_name= True)

    def segmentImage(self, image_path, show_bboxes = False, output_image_name = None):
        
        image = cv2.imread(image_path)
        new_img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # Run detection
        results = self.model.detect([new_img], verbose=1)

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
        if show_bboxes == False:
            
            #apply segmentation mask
            output = display_instances(image, r['rois'], r['masks'], r['class_ids'], class_names)
            
            if output_image_name is not None:
                cv2.imwrite(output_image_name, output)
                print("Processed image saved successfully in your current working directory.")
            return r, output

        else:
            #apply segmentation mask with bounding boxes
            output = display_box_instances(image, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])

            if output_image_name is not None:
                cv2.imwrite(output_image_name, output)
                print("Processed Image saved successfully in your current working directory.")
            return r, output

def random_colors(N):
    np.random.seed(1)
    colors = [tuple(255 * np.random.rand(3)) for _ in range(N)]
    return colors



def apply_mask(input_image, mask, color, alpha=0.8):
    #apply mask to image
    for n, c in enumerate(color):
        input_image[:, :, n] = np.where(
            mask == 1,
            input_image[:, :, n] * (1 - alpha) + alpha * c,
            input_image[:, :, n]
        )
    return input_image

    


def display_instances(image, boxes, masks, class_ids,  class_names):
    
    n_instances = boxes.shape[0]
    colors = random_colors(n_instances)

    if not n_instances:
        print('NO INSTANCES TO DISPLAY')
    else:
        assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]

    for i, color in enumerate(colors):
        mask = masks[:, :, i]

        image = apply_mask(image, mask, color)


    return image





def display_box_instances(image, boxes, masks, class_ids, class_names, scores):
    
    n_instances = boxes.shape[0]
    colors = random_colors(n_instances)

    if not n_instances:
        print('NO INSTANCES TO DISPLAY')
    else:
        assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]

    for i, color in enumerate(colors):
        if not np.any(boxes[i]):
            continue

        y1, x1, y2, x2 = boxes[i]
        label = class_names[class_ids[i]]
        score = scores[i] if scores is not None else None
        caption = '{} {:.2f}'.format(label, score) if score else label
        mask = masks[:, :, i]

        image = apply_mask(image, mask, color)
        image = cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        image = cv2.putText(
            image, caption, (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 0.4, color = (255, 255, 255))

    return image



