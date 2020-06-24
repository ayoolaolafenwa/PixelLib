import cv2
import numpy as np
import random
import os
import sys
import math
from .mask_rcnn import MaskRCNN
from .coco import CocoConfig as inferconfig
import colorsys
import time



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
        print("Processing image...")
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

    



    def segmentFrame(self, frame, show_bboxes = False, output_image_name = None):

        new_img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        print("Processing frame...")
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
            output = display_instances(frame, r['rois'], r['masks'], r['class_ids'], class_names)
            
            if output_image_name is not None:
                cv2.imwrite(output_image_name, output)
                print("Processed image saved successfully in your current working directory.")
            return r, output

        else:
            #apply segmentation mask with bounding boxes
            output = display_box_instances(frame, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])

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
                    output = display_instances(frame, r['rois'], r['masks'], r['class_ids'], class_names)
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
                    output = display_box_instances(frame, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])
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

    def process_camera(self, cam, show_bboxes = False,  output_video_name = None, frames_per_second = None, show_frames = None, frame_name = None, check_fps = False):
        capture = cam
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        codec = cv2.VideoWriter_fourcc(*'DIVX')
        if frames_per_second is not None:
            save_video = cv2.VideoWriter(output_video_name, codec, frames_per_second, (width, height))
        counter = 0
        start = time.time()     

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
                    output = display_instances(frame, r['rois'], r['masks'], r['class_ids'], class_names)
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
            print(f"Processed {counter} frames in {end-start:.1f} seconds")  
            
            if check_fps == True:
                out = capture.get(cv2.CAP_PROP_FPS)
                print(f"{out} frames per seconds")   
           
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
                    output = display_box_instances(frame, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])
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
            print(f"Processed {counter} frames in {end-start:.1f} seconds") 

            if check_fps == True:
                out = capture.get(cv2.CAP_PROP_FPS)
                print(f"{out} frames per seconds")   
        
            capture.release()

            if frames_per_second is not None:
                save_video.release() 

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

    if not n_instances:
        print('NO INSTANCES TO DISPLAY')
    else:
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
        image = cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        image = cv2.putText(
            image, caption, (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 0.4, color = (255, 255, 255))

    return image



