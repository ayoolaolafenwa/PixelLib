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
from datetime import datetime
import imantics
from imantics import Polygons, Mask


class configuration(Config):
    NAME = "configuration"

coco_config = configuration(BACKBONE = "resnet101",  NUM_CLASSES =  81,  class_names = ["BG"], IMAGES_PER_GPU = 1, 
DETECTION_MIN_CONFIDENCE = 0.7,IMAGE_MAX_DIM = 1024, IMAGE_MIN_DIM = 800,IMAGE_RESIZE_MODE ="square",  GPU_COUNT = 1) 


class instance_segmentation():
    def __init__(self, infer_speed = None):
        if infer_speed == "average":
            coco_config.IMAGE_MAX_DIM = 512
            coco_config.IMAGE_MIN_DIM = 512
            coco_config.DETECTION_MIN_CONFIDENCE = 0.45

        elif infer_speed == "fast":
            coco_config.IMAGE_MAX_DIM = 384
            coco_config.IMAGE_MIN_DIM = 384
            coco_config.DETECTION_MIN_CONFIDENCE = 0.25

        elif infer_speed == "rapid":
            coco_config.IMAGE_MAX_DIM = 256
            coco_config.IMAGE_MIN_DIM = 256
            coco_config.DETECTION_MIN_CONFIDENCE = 0.20   
            

        self.model_dir = os.getcwd()

    def load_model(self, model_path):
        self.model = MaskRCNN(mode = "inference", model_dir = self.model_dir, config = coco_config)
        self.model.load_weights(model_path, by_name= True)

    
    def select_target_classes(self,BG = False, person=False, bicycle=False, car=False, motorcycle=False, airplane=False,
                      bus=False, train=False, truck=False, boat=False, traffic_light=False, fire_hydrant=False,
                      stop_sign=False,
                      parking_meter=False, bench=False, bird=False, cat=False, dog=False, horse=False, sheep=False,
                      cow=False, elephant=False, bear=False, zebra=False,
                      giraffe=False, backpack=False, umbrella=False, handbag=False, tie=False, suitcase=False,
                      frisbee=False, skis=False, snowboard=False,
                      sports_ball=False, kite=False, baseball_bat=False, baseball_glove=False, skateboard=False,
                      surfboard=False, tennis_racket=False,
                      bottle=False, wine_glass=False, cup=False, fork=False, knife=False, spoon=False, bowl=False,
                      banana=False, apple=False, sandwich=False, orange=False,
                      broccoli=False, carrot=False, hot_dog=False, pizza=False, donut=False, cake=False, chair=False,
                      couch=False, potted_plant=False, bed=False,
                      dining_table=False, toilet=False, tv=False, laptop=False, mouse=False, remote=False,
                      keyboard=False, cell_phone=False, microwave=False,
                      oven=False, toaster=False, sink=False, refrigerator=False, book=False, clock=False, vase=False,
                      scissors=False, teddy_bear=False, hair_dryer=False,
                      toothbrush=False):

        detected_classes = {}
        target_class_names = [BG, person, bicycle, car, motorcycle, airplane,
                        bus, train, truck, boat, traffic_light, fire_hydrant, stop_sign,
                        parking_meter, bench, bird, cat, dog, horse, sheep, cow, elephant, bear, zebra,
                        giraffe, backpack, umbrella, handbag, tie, suitcase, frisbee, skis, snowboard,
                        sports_ball, kite, baseball_bat, baseball_glove, skateboard, surfboard, tennis_racket,
                        bottle, wine_glass, cup, fork, knife, spoon, bowl, banana, apple, sandwich, orange,
                        broccoli, carrot, hot_dog, pizza, donut, cake, chair, couch, potted_plant, bed,
                        dining_table, toilet, tv, laptop, mouse, remote, keyboard, cell_phone, microwave,
                        oven, toaster, sink, refrigerator, book, clock, vase, scissors, teddy_bear, hair_dryer,
                        toothbrush]
        class_names = ["BG", "person", "bicycle", "car", "motorcycle", "airplane",
                         "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign",
                         "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear",
                         "zebra",
                         "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis",
                         "snowboard",
                         "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
                         "tennis racket",
                         "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
                         "orange",
                         "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
                         "bed",
                         "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
                         "microwave",
                         "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
                         "hair dryer",
                         "toothbrush"]

        for target_class_name, class_name in zip(target_class_names, class_names):
            if (target_class_name == True):
                detected_classes[class_name] = "valid"
            else:
                detected_classes[class_name] = "invalid"
            
        return detected_classes

    def segmentImage(self, image_path, show_bboxes = False, process_frame = False, segment_target_classes = None, extract_segmented_objects = False, 
    save_extracted_objects = False,mask_points_values = False,  output_image_name = None, verbose = None):

        if process_frame ==False:
            image = cv2.imread(image_path)

        else:
            image = image_path

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
        
        """ Code to filter out unused detections and detect specific classes """
        if segment_target_classes is not None:
            bboxes = r['rois']
            scores = r['scores']
            masks = r['masks']
            class_ids = r['class_ids']
            
            
            com_bboxes = []
            com_masks = []
            com_scores = []
            com_class_ids = []
            
            final_dict = []
            for a, b in enumerate(r['class_ids']):
                name = coco_config.class_names[b]

                
                box = bboxes[a]
               
                ma = masks[:, :, a]
                
                score = scores[a]
                
                c_ids = class_ids[a]
                
                
                if (segment_target_classes[name] == "invalid"):
                    continue
                    
                com_bboxes.append(box)
                com_class_ids.append(c_ids)
                com_masks.append(ma)
                com_scores.append(score)
                
                 
            final_bboxes = np.array(com_bboxes)
            
            final_class_ids = np.array(com_class_ids)
            final_masks = np.array(com_masks)
            if len(final_masks != 0):
                final_masks = np.stack(final_masks, axis = 2)
            
            final_scores = np.array(com_scores)
            
            final_dict.append({
                   "rois": final_bboxes,
                   "class_ids": final_class_ids,
                   "scores": final_scores,
                   "masks": final_masks,
                   })
            r = final_dict[0]      
        
            
        if show_bboxes == False:
            output = display_instances(image, r['rois'], r['masks'], r['class_ids'], coco_config.class_names)
            if output_image_name is not None:
                cv2.imwrite(output_image_name, output)
                print("Processed image saved successfully in your current working directory.")   
            
            """ Code to extract and crop out each of the objects segmented in an image """
            if extract_segmented_objects == False:
                
                if mask_points_values == True:
                    mask = r['masks']
                    contain_val = []
                    for a in range(mask.shape[2]):
                        m = mask[:,:,a]
                        mask_values = Mask(m).polygons()
                        val = mask_values.points
                        contain_val.append(val)

                    r['masks'] = contain_val
             

                return r, output



            else:
               

                mask = r['masks']
                m = 0
                for a in range(mask.shape[2]):
                    
                    img = cv2.imread(image_path)
                    
                    for b in range(img.shape[2]):
       
                        img[:,:,b] = img[:,:,b] * mask[:,:,a]
                    m+=1
                    extracted_objects = img[np.ix_(mask[:,:,a].any(1), mask[:,:,a].any(0))]
                    
                    if save_extracted_objects == True:
                        save_path = os.path.join("segmented_object" + "_" + str(m) + ".jpg")
                        cv2.imwrite(save_path, extracted_objects)
        
                
                if mask_points_values == True:
                    mask = r['masks']
                    contain_val = []
                    for a in range(mask.shape[2]):
                        m = mask[:,:,a]
                        mask_values = Mask(m).polygons()
                        val = mask_values.points
                
                        contain_val.append(val)

                    r['masks'] = contain_val


                    extract_mask = extracted_objects
                    object_val = []

                    for a in range(extract_mask.shape[2]):
                        m = extract_mask[:,:,a]
                        mask_values = Mask(m).polygons()
                        val = mask_values.points
                        object_val.append(val)

                    extracted_objects = object_val

            
                """ The mask values of each of the extracted cropped object in the image
                is added to the dictionary containing an array of output values:
                """ 

                r.update({"extracted_objects":extracted_objects})

                return r, output
            
            
        else:
            output = display_box_instances(image, r['rois'], r['masks'], r['class_ids'], coco_config.class_names, r['scores'])

            """ Code to extract and crop out each of the objects segmented in an image """
            if extract_segmented_objects == True:
                mask = r['masks']
                m = 0
                for a in range(mask.shape[2]):
                    
                    img = cv2.imread(image_path)
                    

                    for b in range(img.shape[2]):
       
                        img[:,:,b] = img[:,:,b] * mask[:,:,a]
                    m+=1
                    extracted_objects = img[np.ix_(mask[:,:,a].any(1), mask[:,:,a].any(0))]
                    
                    if save_extracted_objects == True:
                        save_path = os.path.join("segmented_object" + "_" + str(m) + ".jpg")
                        cv2.imwrite(save_path, extracted_objects)
        

                if mask_points_values == True:
                    mask = r['masks']
                    contain_val = []
                    for a in range(mask.shape[2]):
                        m = mask[:,:,a]
                        mask_values = Mask(m).polygons()
                        val = mask_values.points
                
                        contain_val.append(val)

                    r['masks'] = contain_val
                    

                    extract_mask = extracted_objects
                    object_val = []

                    for a in range(extract_mask.shape[2]):
                        m = extract_mask[:,:,a]
                        mask_values = Mask(m).polygons()
                        val = mask_values.points
                        object_val.append(val)

                    extracted_objects = object_val

            

                if output_image_name is not None:
                    cv2.imwrite(output_image_name, output)
                    print("Processed image saved successfully in your current working directory.") 
                
                """ The mask values of each of the extracted cropped object in the image
                is added to the dictionary containing an array of output values:
                """ 
                
                r.update({"extracted_objects":extracted_objects})
                return r, output
            
            else:

                if mask_points_values == True:
                    mask = r['masks']
                    contain_val = []
                    for a in range(mask.shape[2]):
                        m = mask[:,:,a]
                        mask_values = Mask(m).polygons()
                        val = mask_values.points
                        contain_val.append(val)

                    r['masks'] = contain_val

            

                if output_image_name is not None:
                    cv2.imwrite(output_image_name, output)
                    print("Processed image saved successfully in your current working directory.") 

                return r, output    

           
        

    def segmentFrame(self, frame, show_bboxes = False, segment_target_classes = None,mask_points_values = False,  output_image_name = None):
        segmask, output = self.segmentImage(frame, show_bboxes = show_bboxes, process_frame=True, 
        segment_target_classes = segment_target_classes, mask_points_values = mask_points_values,  output_image_name = output_image_name)
        if output_image_name is not None:
            cv2.imwrite(output_image_name, output)
            print("Processed image saved successfully in your current working directory.")

        return segmask, output
            
           
        
            

    def process_video(self, video_path, show_bboxes = False, segment_target_classes = None, mask_points_values = False, 
    output_video_name = None, frames_per_second = None):
        capture = cv2.VideoCapture(video_path)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        codec = cv2.VideoWriter_fourcc(*'DIVX')
        
        if frames_per_second is not None:
            save_video = cv2.VideoWriter(output_video_name, codec, frames_per_second, (width, height))
        counter = 0
        start = time.time()     
           
        
        while True:
            counter +=1
            ret, frame = capture.read()
            if ret:
                #apply segmentation mask
                    
                    
                segmask, output = self.segmentImage(frame, show_bboxes=show_bboxes, segment_target_classes= segment_target_classes,
                process_frame=True, mask_points_values=mask_points_values)
                print("No. of frames:", counter)
                    
                        
                output = cv2.resize(output, (width,height), interpolation=cv2.INTER_AREA)

                if output_video_name is not None:
                    save_video.write(output)

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

        return segmask, output     
 
        

    def process_camera(self, cam, show_bboxes = False, segment_target_classes = None, mask_points_values = False, output_video_name = None, frames_per_second = None, show_frames = None, frame_name = None, verbose = None, check_fps = False):
        capture = cam
        if output_video_name is not None:
          width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
          height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
          save_video = cv2.VideoWriter(output_video_name, cv2.VideoWriter_fourcc(*'DIVX'), frames_per_second, (width, height))
        
        counter = 0
          
        start = datetime.now()       

        
        while True:
                
            ret, frame = capture.read()
            if ret:
                    
                segmask, output = self.segmentImage(frame, show_bboxes=show_bboxes,segment_target_classes= segment_target_classes,
                process_frame=True, mask_points_values=mask_points_values)
                counter += 1 
                        
                output = cv2.resize(output, (width,height), interpolation=cv2.INTER_AREA)

                if show_frames == True:
                    if frame_name is not None:
                        cv2.imshow(frame_name, output)
                            
                        if cv2.waitKey(25) & 0xFF == ord('q'):
                            break  

                if output_video_name is not None:
                    save_video.write(output)

            elif counter == 30:
                break  
                 
            
        end = datetime.now()
        if check_fps == True:
            timetaken = (end-start).total_seconds()
                
            out = counter / timetaken
            print(f"{out:.3f} frames per second")   

        if verbose is not None: 
            print(f"Processed {counter} frames in {timetaken:.1f} seconds")     
           
        capture.release()

        if output_video_name is not None:
            save_video.release()  

        return segmask, output   

        




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
    
    def segmentImage(self, image_path, show_bboxes = False, extract_segmented_objects = False, save_extracted_objects = False,
    mask_points_values = False,  process_frame = False,output_image_name = None, verbose = None):

        if process_frame ==False:
            image = cv2.imread(image_path)

        else:
            image = image_path

        new_img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # Run detection
        if verbose is not None:
            print("Processing image...")
        results = self.model.detect([new_img])    


        r = results[0] 
            
        if show_bboxes == False:
            output = display_instances(image, r['rois'], r['masks'], r['class_ids'], self.config.class_names)
            if output_image_name is not None:
                cv2.imwrite(output_image_name, output)
                print("Processed image saved successfully in your current working directory.")   
            
            """ Code to extract and crop out each of the objects segmented in an image """
                
            if extract_segmented_objects == False:
                
                if mask_points_values == True:
                    mask = r['masks']
                    contain_val = []
                    for a in range(mask.shape[2]):
                        m = mask[:,:,a]
                        mask_values = Mask(m).polygons()
                        val = mask_values.points
                        contain_val.append(val)

                    r['masks'] = contain_val


                return r, output



            else:
               

                mask = r['masks']
                m = 0
                for a in range(mask.shape[2]):
                    if process_frame == False:
                        img = cv2.imread(image_path)
                    else:
                        img = image_path
                    for b in range(img.shape[2]):
       
                        img[:,:,b] = img[:,:,b] * mask[:,:,a]
                    m+=1
                    extracted_objects = img[np.ix_(mask[:,:,a].any(1), mask[:,:,a].any(0))]
                    
                    if save_extracted_objects == True:
                        save_path = os.path.join("segmented_object" + "_" + str(m) + ".jpg")
                        cv2.imwrite(save_path, extracted_objects)
        
                
                if mask_points_values == True:
                    mask = r['masks']
                    contain_val = []
                    for a in range(mask.shape[2]):
                        m = mask[:,:,a]
                        mask_values = Mask(m).polygons()
                        val = mask_values.points
                
                        contain_val.append(val)

                    r['masks'] = contain_val


                    extract_mask = extracted_objects
                    object_val = []

                    for a in range(extract_mask.shape[2]):
                        m = extract_mask[:,:,a]
                        mask_values = Mask(m).polygons()
                        val = mask_values.points
                        object_val.append(val)

                    extracted_objects = object_val

            
                """ The mask values of each of the extracted cropped object in the image
                is added to the dictionary containing an array of output values:
                """ 
                r.update({"extracted_objects":extracted_objects})

                return r, output
            
            
        else:
            output = display_box_instances(image, r['rois'], r['masks'], r['class_ids'], self.config.class_names, r['scores'])

            """ Code to extract and crop out each of the objects segmented in an image """

            if extract_segmented_objects == True:
                mask = r['masks']
                m = 0
                for a in range(mask.shape[2]):
                    if process_frame == False:
                        img = cv2.imread(image_path)
                    else:
                        img = image_path

                    for b in range(img.shape[2]):
       
                        img[:,:,b] = img[:,:,b] * mask[:,:,a]
                    m+=1
                    extracted_objects = img[np.ix_(mask[:,:,a].any(1), mask[:,:,a].any(0))]
                    
                    if save_extracted_objects == True:
                        save_path = os.path.join("segmented_object" + "_" + str(m) + ".jpg")
                        cv2.imwrite(save_path, extracted_objects)
        
               

                if mask_points_values == True:
                    mask = r['masks']
                    contain_val = []
                    for a in range(mask.shape[2]):
                        m = mask[:,:,a]
                        mask_values = Mask(m).polygons()
                        val = mask_values.points
                
                        contain_val.append(val)

                    r['masks'] = contain_val
                    

                    extract_mask = extracted_objects
                    object_val = []

                    for a in range(extract_mask.shape[2]):
                        m = extract_mask[:,:,a]
                        mask_values = Mask(m).polygons()
                        val = mask_values.points
                        object_val.append(val)

                    extracted_objects = object_val

            

                if output_image_name is not None:
                    cv2.imwrite(output_image_name, output)
                    print("Processed image saved successfully in your current working directory.") 
                

                """ The mask values of each of the extracted cropped object in the image
                is added to the dictionary containing an array of output values:
                """
                
                r.update({"extracted_objects":extracted_objects})
                return r, output
            
            else:
                
                if mask_points_values == True:
                    mask = r['masks']
                    contain_val = []
                    for a in range(mask.shape[2]):
                        m = mask[:,:,a]
                        mask_values = Mask(m).polygons()
                        val = mask_values.points
                        contain_val.append(val)

                    r['masks'] = contain_val

            

                if output_image_name is not None:
                    cv2.imwrite(output_image_name, output)
                    print("Processed image saved successfully in your current working directory.") 

                return r, output    

           
        
            
 


    def segmentFrame(self, frame, show_bboxes = False,  mask_points_values = False, output_image_name = None, verbose= None):

            segmask, output = self.segmentImage(frame, show_bboxes=show_bboxes, process_frame=True, mask_points_values=mask_points_values)
            
            if output_image_name is not None:
                cv2.imwrite(output_image_name, output)
                print("Processed image saved successfully in your current working directory.")

            return segmask, output

              

        
    def process_video(self, video_path, show_bboxes = False, mask_points_values = False, output_video_name = None, frames_per_second = None):
        capture = cv2.VideoCapture(video_path)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        codec = cv2.VideoWriter_fourcc(*'DIVX')
        if frames_per_second is not None:
            save_video = cv2.VideoWriter(output_video_name, codec, frames_per_second, (width, height))
        counter = 0
        start = time.time()     
           
        
        while True:
            counter +=1
            ret, frame = capture.read()
            if ret:
                
                segmask, output = self.segmentImage(frame, show_bboxes=show_bboxes,process_frame=True, mask_points_values=mask_points_values)
                print("No. of frames:", counter)
                    
                        
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

        return segmask, output   

        
                  
    def process_camera(self, cam, show_bboxes = False,  mask_points_values = False, output_video_name = None, frames_per_second = None, show_frames = None, frame_name = None, verbose = None, check_fps = False):
        capture = cam
        
        if output_video_name is not None:
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            codec = cv2.VideoWriter_fourcc(*'DIVX')
            save_video = cv2.VideoWriter(output_video_name, codec, frames_per_second, (width, height))

        counter = 0
        start = datetime.now()     

        
        while True:
                
            ret, frame = capture.read()
            if ret:
                    
                segmask, output = self.segmentImage(frame, show_bboxes=False,process_frame=True, mask_points_values=mask_points_values)

                        
                output = cv2.resize(output, (width,height), interpolation=cv2.INTER_AREA)

                if show_frames == True:
                    if frame_name is not None:
                        cv2.imshow(frame_name, output)
                            
                        if cv2.waitKey(25) & 0xFF == ord('q'):
                            break  

                if output_video_name is not None:
                    save_video.write(output)

                    
                   
            elif counter == 30:
                break  
                 
        end = datetime.now() 
            
            
        if check_fps == True:
            timetaken = (end-start).total_seconds()
            fps = counter/timetaken
            print(f"{fps} frames per seconds")   

        if verbose is not None:
            print(f"Processed {counter} frames in {timetaken:.1f} seconds") 
                        
           
        capture.release()

        if output_video_name is not None:
            save_video.release()  
   

        return segmask, output     

        



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



