import cv2
import numpy as np
import random
import os
import sys
import math
from pixellib.mask_rcnn import MaskRCNN
from pixellib.config import Config
import colorsys
import time

class configuration(Config):
    NAME = "configuration"

coco_config = configuration(BACKBONE = "resnet101",  NUM_CLASSES =  81,  class_names = ["BG"], IMAGES_PER_GPU = 1, 
DETECTION_MIN_CONFIDENCE = 0.7,IMAGE_MAX_DIM = 1024, IMAGE_MIN_DIM = 800,IMAGE_RESIZE_MODE ="square",  GPU_COUNT = 1) 


class instance_segmentation():
    def __init__(self, detection_speed = None):
        if detection_speed == "average":
            coco_config.IMAGE_MAX_DIM = 512
            coco_config.IMAGE_MIN_DIM = 512
            coco_config.DETECTION_MIN_CONFIDENCE = 0.45

        elif detection_speed == "fast":
            coco_config.IMAGE_MAX_DIM = 384
            coco_config.IMAGE_MIN_DIM = 384
            coco_config.DETECTION_MIN_CONFIDENCE = 0.25

        elif detection_speed == "rapid":
            coco_config.IMAGE_MAX_DIM = 256
            coco_config.IMAGE_MIN_DIM = 256
            coco_config.DETECTION_MIN_CONFIDENCE = 0.20   
            

        self.model_dir = os.getcwd()

    def load_model(self, model_path):
        self.model = MaskRCNN(mode = "inference", model_dir = self.model_dir, config = coco_config)
        self.model.load_weights(model_path, by_name= True)


    def segmentImage(self, image_path, show_bboxes = False,  output_image_name = None, verbose = None):
        
        image = cv2.imread(image_path)
        new_img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # Run detection
        if verbose is not None:
            print("Processing image...")
        results = self.model.detect([new_img])    

        coco_config.class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
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
            output = display_instances(image, r['rois'], r['masks'], r['class_ids'], coco_config.class_names)
            
            if output_image_name is not None:
                cv2.imwrite(output_image_name, output)
                print("Processed image saved successfully in your current working directory.")
            return r, output

        else:
            #apply segmentation mask with bounding boxes
            output = display_box_instances(image, r['rois'], r['masks'], r['class_ids'], coco_config.class_names, r['scores'])

            if output_image_name is not None:
                cv2.imwrite(output_image_name, output)
                print("Processed Image saved successfully in your current working directory.")
            return r, output

    



    def segmentFrame(self, frame, show_bboxes = False, output_image_name = None, verbose = None):

        new_img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        if verbose is not None:
            print("Processing frame...")
        # Run detection
        results = self.model.detect([new_img])

        coco_config.class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
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
            output = display_instances(frame, r['rois'], r['masks'], r['class_ids'], coco_config.class_names)
            
            if output_image_name is not None:
                cv2.imwrite(output_image_name, output)
                print("Processed image saved successfully in your current working directory.")
            return r, output

        else:
            #apply segmentation mask with bounding boxes
            output = display_box_instances(frame, r['rois'], r['masks'], r['class_ids'], coco_config.class_names, r['scores'])

            if output_image_name is not None:
                cv2.imwrite(output_image_name, output)
                print("Processed Image saved successfully in your current working directory.")
            return r, output
        

    def process_video(self, video_path, show_bboxes = False,  output_video_name = None, frames_per_second = None):
        capture = cv2.VideoCapture(video_path)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        codec = cv2.VideoWriter_fourcc(*'DIVX')
        coco_config.class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
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
        if frames_per_second is not None:
            save_video = cv2.VideoWriter(output_video_name, codec, frames_per_second, (width, height))
        counter = 0
        start = time.time()     
           
        if show_bboxes == False:
            while True:
                counter +=1
                ret, frame = capture.read()
                if ret:
                    # Run detection
                    results = self.model.detect([frame])
                    print("No. of frames:", counter)
                    r = results[0]   
                    #apply segmentation mask
                    output = display_instances(frame, r['rois'], r['masks'], r['class_ids'], coco_config.class_names)
                    output = cv2.resize(output, (width,height), interpolation=cv2.INTER_AREA)

                    if output_video_name is not None:
                        save_video.write(output)
                   
                else:
                    break 
                  
            end = time.time() 
            print(f"Processed {counter} frames in {end-start:.1f} seconds")  
            
           
            capture.release()
            if frames_per_second is not None:
                save_video.release()    
            return r, output   

        else:
            while True:
                counter +=1
                ret, frame = capture.read()
                if ret:
                    # Run detection
                    results = self.model.detect([frame])
                    print("No. of frames:", counter)
                    r = results[0]   
                    #apply segmentation mask with bounding boxes
                    output = display_box_instances(frame, r['rois'], r['masks'], r['class_ids'], coco_config.class_names, r['scores'])
                    output = cv2.resize(output, (width,height), interpolation=cv2.INTER_AREA)
        
                    if output_video_name is not None:
                        save_video.write(output)
                else:
                    break
            
            capture.release()

            end = time.time()
            print(f"Processed {counter} frames in {end-start:.1f} seconds")  
        
            
            if frames_per_second is not None:
                save_video.release()
                 
            return r, output         


    def process_camera(self, cam, show_bboxes = False,  output_video_name = None, frames_per_second = None, show_frames = None, frame_name = None, verbose = None, check_fps = False):
        capture = cam
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        codec = cv2.VideoWriter_fourcc(*'DIVX')
        if frames_per_second is not None:
            save_video = cv2.VideoWriter(output_video_name, codec, frames_per_second, (width, height))
        counter = 0
        start = time.time()     

        coco_config.class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
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
           
        if show_bboxes == False:
            while True:
                counter +=1
                ret, frame = capture.read()
                if ret:
                    # Run detection
                    results = self.model.detect([frame])
                    if verbose is not None:
                        print("No. of frames:", counter)
                    r = results[0]   
                    #apply segmentation mask
                    output = display_instances(frame, r['rois'], r['masks'], r['class_ids'], coco_config.class_names)
                    output = cv2.resize(output, (width,height), interpolation=cv2.INTER_AREA)

                    if show_frames == True:
                        if frame_name is not None:
                            cv2.imshow(frame_name, output)
                            
                            if cv2.waitKey(25) & 0xFF == ord('q'):
                                break  

                    if output_video_name is not None:
                        save_video.write(output)
                   
                else:
                    break  
                 
            end = time.time()
            if verbose is not None: 
                print(f"Processed {counter} frames in {end-start:.1f} seconds")  
            
            if check_fps == True:
                out = capture.get(cv2.CAP_PROP_FPS)
                print(f"{out} frames per second")   
           
            capture.release()

            if frames_per_second is not None:
                save_video.release()  

             

            return r, output     

        else:
            while True:
                counter +=1
                ret, frame = capture.read()
                if ret:
                    # Run detection
                    results = self.model.detect([frame])
                    if verbose is not None:
                        print("No. of frames:", counter)
                    r = results[0]   
                    #apply segmentation mask with bounding boxes
                    output = display_box_instances(frame, r['rois'], r['masks'], r['class_ids'], coco_config.class_names, r['scores'])
                    output = cv2.resize(output, (width,height), interpolation=cv2.INTER_AREA)

                    if show_frames == True:
                        if frame_name is not None:
                            cv2.imshow(frame_name, output)

                            if cv2.waitKey(25) & 0xFF == ord('q'):
                                break  
        
                    if output_video_name is not None:
                        save_video.write(output)
                else:
                    break
            end = time.time()
            if verbose is not None:
                print(f"Processed {counter} frames in {end-start:.1f} seconds") 

            if check_fps == True:
                out = capture.get(cv2.CAP_PROP_FPS)
                print(f"{out} frames per second")   
        
            capture.release()

            if frames_per_second is not None:
                save_video.release() 

            return r, output          










#############################################################
#############################################################
""" CLASS FOR PERFORMING INFERENCE WITH A CUSTOM MODEL """
#############################################################
#############################################################




class custom_segmentation:
    def __init__(self):
       self.model_dir = os.getcwd()

    def inferConfig(self,name = None, network_backbone = "resnet101",  num_classes =  1,  class_names = ["BG"], batch_size = 1, detection_threshold = 0.7, 
    image_max_dim = 512, image_min_dim = 512, image_resize_mode ="square", gpu_count = 1):
        self.config = Config(BACKBONE = network_backbone, NUM_CLASSES = 1 +  num_classes,  class_names = class_names, 
        IMAGES_PER_GPU = batch_size, IMAGE_MAX_DIM = image_max_dim, IMAGE_MIN_DIM = image_min_dim, DETECTION_MIN_CONFIDENCE = detection_threshold,
        IMAGE_RESIZE_MODE = image_resize_mode,GPU_COUNT = gpu_count)
        
    def load_model(self, model_path):
        #load the weights for COCO
        self.model = MaskRCNN(mode="inference", model_dir = self.model_dir, config=self.config)
        self.model.load_weights(model_path, by_name=True)
    
    def segmentImage(self, image_path, show_bboxes = False, output_image_name = None, verbose = None):
        image = cv2.imread(image_path)
        new_img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # Run detection
        if verbose is not None:
            print("Processing image...")
        
        results = self.model.detect([new_img])

    
        r = results[0]       
        if show_bboxes == False:
            
            #apply segmentation mask
            output = display_instances(image, r['rois'], r['masks'], r['class_ids'],self.config.class_names)
            
            if output_image_name is not None:
                cv2.imwrite(output_image_name, output)
                print("Processed image saved successfully in your current working directory.")
            return r, output

        else:
            #apply segmentation mask with bounding boxes
            output = display_box_instances(image, r['rois'], r['masks'], r['class_ids'], self.config.class_names, r['scores'])

            if output_image_name is not None:
                cv2.imwrite(output_image_name, output)
                print("Processed Image saved successfully in your current working directory.")
    
            return r, output   


    def segmentFrame(self, frame, show_bboxes = False, output_image_name = None, verbose= None):

        new_img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        if verbose is not None:
            print("Processing frame...")
        # Run detection
        results = self.model.detect([new_img])

        r = results[0]  
    
        if show_bboxes == False:
            
            #apply segmentation mask
            output = display_instances(frame, r['rois'], r['masks'], r['class_ids'], self.config.class_names)
            
            if output_image_name is not None:
                cv2.imwrite(output_image_name, output)
                print("Processed image saved successfully in your current working directory.")
            return r, output

        else:
            #apply segmentation mask with bounding boxes
            output = display_box_instances(frame, r['rois'], r['masks'], r['class_ids'], self.config.class_names, r['scores'])

            if output_image_name is not None:
                cv2.imwrite(output_image_name, output)
                print("Processed Image saved successfully in your current working directory.")
            return r, output
        
    def process_video(self, video_path, show_bboxes = False,  output_video_name = None, frames_per_second = None):
        capture = cv2.VideoCapture(video_path)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        codec = cv2.VideoWriter_fourcc(*'DIVX')
        if frames_per_second is not None:
            save_video = cv2.VideoWriter(output_video_name, codec, frames_per_second, (width, height))
        counter = 0
        start = time.time()     
           
        if show_bboxes == False:
            while True:
                counter +=1
                ret, frame = capture.read()
                if ret:
                    # Run detection
                    results = self.model.detect([frame], verbose=0)
                    print("No. of frames:", counter)
                    r = results[0]   
                    #apply segmentation mask
                    output = display_instances(frame, r['rois'], r['masks'], r['class_ids'], self.config.class_names)
                    output = cv2.resize(output, (width,height), interpolation=cv2.INTER_AREA)

                    if output_video_name is not None:
                        save_video.write(output)
                   
                else:
                    break 
                  
            end = time.time() 
            print(f"Processed {counter} frames in {end-start:.1f} seconds")  
            
           
            capture.release()
            if frames_per_second is not None:
                save_video.release()    
            return r, output   

        else:
            while True:
                counter +=1
                ret, frame = capture.read()
                if ret:
                    # Run detection
                    results = self.model.detect([frame], verbose=0)
                    print("No. of frames:", counter)
                    r = results[0]   
                    #apply segmentation mask with bounding boxes
                    output = display_box_instances(frame, r['rois'], r['masks'], r['class_ids'], self.config.class_names, r['scores'])
                    output = cv2.resize(output, (width,height), interpolation=cv2.INTER_AREA)
        
                    if output_video_name is not None:
                        save_video.write(output)
                else:
                    break
            
            capture.release()

            end = time.time()
            print(f"Processed {counter} frames in {end-start:.1f} seconds")  
        
            
            if frames_per_second is not None:
                save_video.release()
                 
            return r, output         
    def process_camera(self, cam, show_bboxes = False,  output_video_name = None, frames_per_second = None, show_frames = None, frame_name = None, verbose = None, check_fps = False):
        capture = cam
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        codec = cv2.VideoWriter_fourcc(*'DIVX')
        if frames_per_second is not None:
            save_video = cv2.VideoWriter(output_video_name, codec, frames_per_second, (width, height))
        counter = 0
        start = time.time()     

        if show_bboxes == False:
            while True:
                counter +=1
                ret, frame = capture.read()
                if ret:
                    # Run detection
                    results = self.model.detect([frame])
                    if verbose is not None:
                       print("No. of frames:", counter)
                    r = results[0]   
                    #apply segmentation mask
                    output = display_instances(frame, r['rois'], r['masks'], r['class_ids'], self.config.class_names)
                    output = cv2.resize(output, (width,height), interpolation=cv2.INTER_AREA)

                    if show_frames == True:
                        if frame_name is not None:
                            cv2.imshow(frame_name, output)
                            
                            if cv2.waitKey(25) & 0xFF == ord('q'):
                                break  

                    if output_video_name is not None:
                        save_video.write(output)
                   
                else:
                    break  
                 
            end = time.time() 
            if verbose is not None:
                print(f"Processed {counter} frames in {end-start:.1f} seconds")  
            
            if check_fps == True:
                out = capture.get(cv2.CAP_PROP_FPS)
                print(f"{out} frames per second")   
           
            capture.release()

            if frames_per_second is not None:
                save_video.release()  

             

            return r, output     

        else:
            while True:
                counter +=1
                ret, frame = capture.read()
                if ret:
                    # Run detection
                    results = self.model.detect([frame])
                    if verbose is not None:
                        print("No. of frames:", counter)
                    r = results[0]   
                    #apply segmentation mask with bounding boxes
                    output = display_box_instances(frame, r['rois'], r['masks'], r['class_ids'], self.config.class_names, r['scores'])
                    output = cv2.resize(output, (width,height), interpolation=cv2.INTER_AREA)

                    if show_frames == True:
                        if frame_name is not None:
                            cv2.imshow(frame_name, output)

                            if cv2.waitKey(25) & 0xFF == ord('q'):
                                break  
        
                    if output_video_name is not None:
                        save_video.write(output)
                else:
                    break
            end = time.time()
            if verbose is not None:
                print(f"Processed {counter} frames in {end-start:.1f} seconds") 

            if check_fps == True:
                out = capture.get(cv2.CAP_PROP_FPS)
                print(f"{out} frames per seconds")   
        
            capture.release()

            if frames_per_second is not None:
                save_video.release() 

            return r, output          





################VISUALIZATION CODE ##################




def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors


def apply_mask(image, mask, color, alpha=0.5):
    """Apply the given mask to the image.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    return image

    


def display_instances(image, boxes, masks, class_ids,  class_name):
    
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





def display_box_instances(image, boxes, masks, class_ids, class_name, scores):
    
    n_instances = boxes.shape[0]
    colors = random_colors(n_instances)

    
    assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]

    for i, color in enumerate(colors):
        if not np.any(boxes[i]):
            continue

        y1, x1, y2, x2 = boxes[i]
        label = class_name[class_ids[i]]
        score = scores[i] if scores is not None else None
        caption = '{} {:.2f}'.format(label, score) if score else label
        mask = masks[:, :, i]

        image = apply_mask(image, mask, color)
        color_rec = [int(c) for c in np.array(colors[i]) * 255]
        image = cv2.rectangle(image, (x1, y1), (x2, y2), color_rec, 2)
        image = cv2.putText(
            image, caption, (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 0.5, color = (255, 255, 255))

    return image



