import cv2
import numpy as np
import random
import os
import math
from .mask_rcnn import MaskRCNN
from .config import Config
import colorsys
import time
from datetime import datetime
import imantics
from imantics import Polygons, Mask
import tensorflow as tf
from pathlib import Path

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

        self.model_dir = os.getcwd()

    def load_model(self, model_path, confidence= None):

        if confidence is not None:
            coco_config.DETECTION_MIN_CONFIDENCE = confidence

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


    def filter_objects(self, segvalues, segment_target_classes):
        """ Code to filter out unused detections and detect specific classes """
        
        bboxes = segvalues['rois']
        scores = segvalues['scores']
        masks = segvalues['masks']
        class_ids = segvalues['class_ids']
            
            
        com_bboxes = []
        com_masks = []
        com_scores = []
        com_class_ids = []
            
        final_dict = []
        for a, b in enumerate(segvalues['class_ids']):
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
            "masks": final_masks,})
        
        final_values = final_dict[0]   
        
        
        return final_values
            


    
    def segmentImage(self, image_path, show_bboxes = False,  segment_target_classes = None, extract_segmented_objects = False, 
    save_extracted_objects = False,mask_points_values = False,  output_image_name = None,  text_thickness = 0,
    text_size = 0.6, box_thickness = 2, verbose = None):
        image = cv2.imread(image_path)

        new_img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # Run detection
        if verbose is not None:
            print("Processing image...")
        results = self.model.detect([new_img])    


        r = results[0]

        """Filter unused detections and detect specific classes """
        
        if segment_target_classes is not None:
            r = self.filter_objects(r, segment_target_classes) 
        
            
        if show_bboxes == False:

            output = display_instances(image, r['rois'], r['masks'], r['class_ids'], coco_config.class_names)

        else: 
            output = display_box_instances(image, r['rois'], r['masks'], r['class_ids'], coco_config.class_names, r['scores'], 
        text_size = text_size, box_thickness=box_thickness, text_thickness=text_thickness)   

        if output_image_name is not None:
            cv2.imwrite(output_image_name, output)
            print("Processed image saved successfully in your current working directory.")   


            
        
        if extract_segmented_objects == False:
                
            if mask_points_values == True:
                mask = r['masks']
                contain_val = []
                    
                for a in range(mask.shape[2]):
                    m = mask[:,:,a]
                    mask_values = Mask(m).polygons()
                    val = mask_values.points
                    contain_val.append(val)
                contain_val = np.asarray(contain_val, dtype = object)

                r['masks'] = contain_val
             

            return r, output

        else:
               
            """ Code to extract and crop out each of the objects segmented in an image """

            mask = r['masks']
            m = 0
            ex = []
            if len(mask != 0):
                for a in range(mask.shape[2]):
                    
                    img = cv2.imread(image_path)
                    
                    for b in range(img.shape[2]):
       
                        img[:,:,b] = img[:,:,b] * mask[:,:,a]
                    m+=1
                    extracted_objects = img[np.ix_(mask[:,:,a].any(1), mask[:,:,a].any(0))]
                    
                    ex.append(extracted_objects)
                    if save_extracted_objects == True:
                        save_path = os.path.join("segmented_object" + "_" + str(m) + ".jpg")
                        cv2.imwrite(save_path, extracted_objects)

                extracted_objects = np.array(ex, dtype = object)
                
                
                if mask_points_values == True:
                    mask = r['masks']
                    contain_val = []
                    for a in range(mask.shape[2]):
                        m = mask[:,:,a]
                        mask_values = Mask(m).polygons()
                        val = mask_values.points
                
                        contain_val.append(val)
                    contain_val = np.array(contain_val, dtype= object)


                    r['masks'] = contain_val


                    extract_mask = extracted_objects
                    object_val = []

                    for a in range(extract_mask.shape[2]):
                        m = extract_mask[:,:,a]
                        mask_values = Mask(m).polygons()
                        val = mask_values.points
                        object_val.append(val)

                    object_val = np.array(object_val)


                    extracted_objects = object_val

            
                """ The mask values of each of the extracted cropped object in the image
                is added to the dictionary containing an array of output values:
                """ 
                r.update({"extracted_objects":extracted_objects})

                return r, output       

            
           

    def segmentBatch(self,input_folder, show_bboxes = False, segment_target_classes = None, extract_segmented_objects = False, 
    save_extracted_objects = False,mask_points_values = False,  output_folder_name = None,  text_thickness = 0,
    text_size = 0.6, box_thickness = 2,  verbose = None):

        if output_folder_name is not None:
            if not os.path.exists(output_folder_name):
                os.mkdir(output_folder_name)

        res = []
        out = []


        
        for p in Path(input_folder).glob('*'): 
            path = str(p)
            
            if extract_segmented_objects == False:
                for name in [".jpg", ".png", ".tif"]:
                    
                
                    if os.path.abspath(p).endswith(name):
                        path = str(p)
                        results, output = self.segmentImage(path, show_bboxes = show_bboxes,
                        segment_target_classes = segment_target_classes, text_thickness = text_thickness,text_size = text_size,
                        box_thickness = box_thickness, mask_points_values = mask_points_values, verbose = verbose)

                        if output_folder_name is not None:
                            path = str(p)
                            n, ext = os.path.splitext(path)
                            name = os.path.basename(path)
                            name = '.'.join(name.split('.')[:-1]) + ext
                            output_path = os.path.join(output_folder_name, name)
                        
                            cv2.imwrite(output_path, output)  
    
                
            else:
                for name in [".jpg", ".png", ".tif"]:
                
                
                    if os.path.abspath(p).endswith(name):
                        path = str(p)
                
                        results, output = self.segmentImage(path, show_bboxes = show_bboxes,
                        segment_target_classes = segment_target_classes, text_thickness = text_thickness,text_size = text_size, 
                        box_thickness = box_thickness, mask_points_values = mask_points_values, verbose = verbose)

                        if output_folder_name is not None:
                            path = str(p)
                            n, ext = os.path.splitext(path)
                            name = os.path.basename(path)
                            name = '.'.join(name.split('.')[:-1]) + ext
                            output_path = os.path.join(output_folder_name, name)
                            cv2.imwrite(output_path, output)  

                        """Code to extract and crop out each of the objects segmented in an image """

                        mask = results['masks']
                        m = 0
                        ex = []
                        if len(mask != 0):
                            for a in range(mask.shape[2]):
                    
                                img = cv2.imread(path)
                    
                                for b in range(img.shape[2]):
       
                                    img[:,:,b] = img[:,:,b] * mask[:,:,a]
                                m+=1
                                extracted_objects = img[np.ix_(mask[:,:,a].any(1), mask[:,:,a].any(0))]
                    
                                ex.append(extracted_objects)
                                if save_extracted_objects == True:
                             
                                    name, ext = os.path.splitext(path) 
                                    dir_extracts = os.path.join(name + "_" + "extracts")  
                              
                            
                                    if not os.path.exists(dir_extracts):
                                        os.mkdir(dir_extracts)
                                
                                    save_path = os.path.join("segmented_object" + "_" + str(m) + ".jpg")    
                                    n, ext = os.path.splitext(save_path)
                                    n = os.path.basename(save_path)
                                    n = '.'.join(n.split('.')[:-1]) + ext
                                    output_path = os.path.join(dir_extracts, n)
                            
                            
                                    cv2.imwrite(output_path, extracted_objects)
                            

                                extracted_objects = np.array(ex, dtype = object)
                
                
                            if mask_points_values == True:
                                mask = results['masks']
                                contain_val = []
                                for a in range(mask.shape[2]):
                                    m = mask[:,:,a]
                                    mask_values = Mask(m).polygons()
                                    val = mask_values.points
                
                                    contain_val.append(val)
                                contain_val = np.array(contain_val, dtype= object)


                                results['masks'] = contain_val


                                extract_mask = extracted_objects
                                object_val = []
                            
                                
            
                            """ The mask values of each of the extracted cropped object in the image
                            is added to the dictionary containing an array of output values:
                            """ 
                            results.update({"extracted_objects":extracted_objects})
                           
            res.append(results)
            
            out.append(output)  


        return res, out   



    def segmentFrame(self, frame, show_bboxes = False,  segment_target_classes = None, extract_segmented_objects = False,
    text_thickness = 0,text_size = 0.6, box_thickness = 2, save_extracted_objects = False,mask_points_values = False,  
    output_image_name = None, verbose = None):


        new_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        # Run detection
        if verbose is not None:
            print("Processing image...")
        results = self.model.detect([new_frame])    


        r = results[0] 

        """Filter unused detections and detect specific classes """
        
        if segment_target_classes is not None:
            r = self.filter_objects(r, segment_target_classes) 
        
            
        if show_bboxes == False:

            output = display_instances(frame, r['rois'], r['masks'], r['class_ids'], coco_config.class_names)

        else: 
            output = display_box_instances(frame, r['rois'], r['masks'], r['class_ids'], coco_config.class_names, r['scores'],
            text_size = text_size, box_thickness=box_thickness, text_thickness=text_thickness)   

        if output_image_name is not None:
            cv2.imwrite(output_image_name, output)
            print("Processed image saved successfully in your current working directory.")   


            
        
        if extract_segmented_objects == False:
                
            if mask_points_values == True:
                mask = r['masks']
                contain_val = []
                    
                for a in range(mask.shape[2]):
                    m = mask[:,:,a]
                    mask_values = Mask(m).polygons()
                    val = mask_values.points
                    contain_val.append(val)

                contain_val = np.asarray(contain_val, dtype = object)
                r['masks'] = contain_val
             

            return r, output

        else:

            """ Code to extract and crop out each of the objects segmented in an image """
        
            mask = r['masks']
            m = 0
            ex = []
            if len(mask != 0):
                for a in range(mask.shape[2]):
                    
                    ori_frame = new_frame
                    img = cv2.cvtColor(ori_frame, cv2.COLOR_RGB2BGR)
                
                    
                    for b in range(img.shape[2]):
       
                        img[:,:,b] = img[:,:,b] * mask[:,:,a]
                    m+=1
                    extracted_objects = img[np.ix_(mask[:,:,a].any(1), mask[:,:,a].any(0))]
                    ex.append(extracted_objects)
                    if save_extracted_objects == True:
                        save_path = os.path.join("segmented_object" + "_" + str(m) + ".jpg")
                        cv2.imwrite(save_path, extracted_objects)

                extracted_objects = np.array(ex, dtype = object)


                if mask_points_values == True:
                        
                    contain_val = []
                    for a in range(mask.shape[2]):
                        m = mask[:,:,a]
                        mask_values = Mask(m).polygons()
                        val = mask_values.points
                
                        contain_val.append(val)

                    contain_val = np.asarray(contain_val, dtype = object)    
                    r['masks'] = contain_val

                       
                    extract_mask = extracted_objects
                    object_val = []

                    for a in range(extract_mask.shape[2]):
                        m = extract_mask[:,:,a]
                        mask_values = Mask(m).polygons()
                        val = mask_values.points
                        object_val.append(val)

                    object_val = np.asarray(object_val, dtype = object)
                    extracted_objects = object_val

                """ The mask values of each of the extracted cropped object in the image
                is added to the dictionary containing an array of output values:
                """ 

                r.update({"extracted_objects": extracted_objects})
                
            return r, output
        

        

    def process_video(self, video_path, show_bboxes = False, segment_target_classes = None, extract_segmented_objects = False, save_extracted_objects = False,
    text_thickness = 0,text_size = 0.6, box_thickness = 2, mask_points_values = False, output_video_name = None, frames_per_second = None):
        capture = cv2.VideoCapture(video_path)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        codec = cv2.VideoWriter_fourcc(*'DIVX')
        if frames_per_second is not None:
            save_video = cv2.VideoWriter(output_video_name, codec, frames_per_second, (width, height))
        counter = 0
        start = time.time()     
           
        while True:
                
            ret, frame = capture.read()
            counter +=1

            if ret:
                
                    
                seg, output =  self.segmentFrame(frame, show_bboxes=show_bboxes, segment_target_classes=segment_target_classes,
                        text_thickness = text_thickness,text_size = text_size, box_thickness = box_thickness,
                        extract_segmented_objects=extract_segmented_objects, save_extracted_objects=save_extracted_objects,
                        mask_points_values= mask_points_values )
                    
                print("No. of frames:", counter)   
                output = cv2.resize(output, (width,height), interpolation=cv2.INTER_AREA)

                

                if output_video_name is not None:
                    save_video.write(output)

                  
                   
            else:
                break  
                 
        end = time.time() 
            
            
        print(f"Processed {counter} frames in {end-start:.1f} seconds")  
           
        capture.release()

        if output_video_name is not None:
            save_video.release()  
   
        
        return seg, output     
            

    
    def process_camera(self, cam, show_bboxes = False, segment_target_classes = None, extract_segmented_objects = False, save_extracted_objects = False,  
    text_thickness = 0,text_size = 0.6, box_thickness = 2,mask_points_values = False, output_video_name = None, frames_per_second = None,
     show_frames = None, frame_name = None, verbose = None, check_fps = False):
        capture = cam
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        if output_video_name is not None:
            codec = cv2.VideoWriter_fourcc(*'DIVX')
            save_video = cv2.VideoWriter(output_video_name, codec, frames_per_second, (width, height))

        counter = 0
        start = datetime.now()     

        
        while True:
                
            ret, frame = capture.read()
            if ret:
                seg, output =  self.segmentFrame(frame, show_bboxes=show_bboxes, segment_target_classes=segment_target_classes,
                        text_thickness = text_thickness,text_size = text_size, box_thickness = box_thickness,
                        extract_segmented_objects=extract_segmented_objects, save_extracted_objects=save_extracted_objects,
                        mask_points_values= mask_points_values )
                    
                        
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
   
        
        return seg, output     
    

    


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
    
    

    def segmentImage(self, image_path, show_bboxes = False,   extract_segmented_objects = False, 
    save_extracted_objects = False,mask_points_values = False,  output_image_name = None,
    text_thickness = 0,text_size = 0.6, box_thickness = 2, verbose = None):

        
        image = cv2.imread(image_path)

        new_img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # Run detection
        if verbose is not None:
            print("Processing image...")
        results = self.model.detect([new_img])    


        r = results[0] 
        
            
        if show_bboxes == False:

            output = display_instances(image, r['rois'], r['masks'], r['class_ids'], self.config.class_names)

        else: 
            output = display_box_instances(image, r['rois'], r['masks'], r['class_ids'], self.config.class_names, r['scores'],
            text_thickness = text_thickness,text_size = text_size, box_thickness = box_thickness)   

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
                contain_val = np.asarray(contain_val, dtype = object)

                r['masks'] = contain_val
             

            return r, output

        else:
               
            ex = []    
            mask = r['masks']
            m = 0
            if len(mask != 0):
                for a in range(mask.shape[2]):
                    
                    img = cv2.imread(image_path)
                    
                    for b in range(img.shape[2]):
       
                        img[:,:,b] = img[:,:,b] * mask[:,:,a]
                    m+=1
                    extracted_objects = img[np.ix_(mask[:,:,a].any(1), mask[:,:,a].any(0))]
                    ex.append(extracted_objects)
                    if save_extracted_objects == True:
                        save_path = os.path.join("segmented_object" + "_" + str(m) + ".jpg")
                        cv2.imwrite(save_path, extracted_objects)
        
                extracted_objects = np.array(ex, dtype=object)

                if mask_points_values == True:
                    mask = r['masks']
                    contain_val = []
                    for a in range(mask.shape[2]):
                        m = mask[:,:,a]
                        mask_values = Mask(m).polygons()
                        val = mask_values.points
                
                        contain_val.append(val)
                    contain_val = np.array(contain_val, dtype= object)


                    r['masks'] = contain_val


                    extract_mask = extracted_objects
                    object_val = []

                    for a in range(extract_mask.shape[2]):
                        m = extract_mask[:,:,a]
                        mask_values = Mask(m).polygons()
                        val = mask_values.points
                        object_val.append(val)
                    object_val = np.array(object_val)


                    extracted_objects = object_val

            
                """ The mask values of each of the extracted cropped object in the image
                is added to the dictionary containing an array of output values:
                """ 
                r.update({"extracted_objects":extracted_objects})

                return r, output



    def segmentBatch(self,input_folder, show_bboxes = False,  extract_segmented_objects = False, 
    save_extracted_objects = False,mask_points_values = False,  output_folder_name = None,  text_thickness = 0,
    text_size = 0.6, box_thickness = 2,  verbose = None):

        if output_folder_name is not None:
            if not os.path.exists(output_folder_name):
                os.mkdir(output_folder_name)

        res = []
        out = []


        
        for p in Path(input_folder).glob('*'): 
            path = str(p)
            
            if extract_segmented_objects == False:
                for name in [".jpg", ".png", ".tif"]:
                    
                
                    if os.path.abspath(p).endswith(name):
                        path = str(p)
                        results, output = self.segmentImage(path, show_bboxes = show_bboxes,
                        text_thickness = text_thickness,text_size = text_size,
                        box_thickness = box_thickness, mask_points_values = mask_points_values, verbose = verbose)

                        if output_folder_name is not None:
                            path = str(p)
                            n, ext = os.path.splitext(path)
                            name = os.path.basename(path)
                            name = '.'.join(name.split('.')[:-1]) + ext
                            output_path = os.path.join(output_folder_name, name)
                        
                            cv2.imwrite(output_path, output)  

            else:
                for name in [".jpg", ".png", ".tif"]:
                
                
                    if os.path.abspath(p).endswith(name):
                        path = str(p)
                
                        results, output = self.segmentImage(path, show_bboxes = show_bboxes,
                        text_thickness = text_thickness,text_size = text_size, 
                        box_thickness = box_thickness, mask_points_values = mask_points_values, verbose = verbose)

                        if output_folder_name is not None:
                            path = str(p)
                            n, ext = os.path.splitext(path)
                            name = os.path.basename(path)
                            name = '.'.join(name.split('.')[:-1]) + ext
                            output_path = os.path.join(output_folder_name, name)
                            cv2.imwrite(output_path, output)  

                        """Code to extract and crop out each of the objects segmented in an image """

                        mask = results['masks']
                        m = 0
                        ex = []
                        if len(mask != 0):
                            for a in range(mask.shape[2]):
                    
                                img = cv2.imread(path)
                    
                                for b in range(img.shape[2]):
       
                                    img[:,:,b] = img[:,:,b] * mask[:,:,a]
                                m+=1
                                extracted_objects = img[np.ix_(mask[:,:,a].any(1), mask[:,:,a].any(0))]
                    
                                ex.append(extracted_objects)
                                if save_extracted_objects == True:
                             
                                    name, ext = os.path.splitext(path) 
                                    dir_extracts = os.path.join(name + "_" + "extracts")  
                              
                            
                                    if not os.path.exists(dir_extracts):
                                        os.mkdir(dir_extracts)
                                
                                    save_path = os.path.join("segmented_object" + "_" + str(m) + ".jpg")    
                                    n, ext = os.path.splitext(save_path)
                                    n = os.path.basename(save_path)
                                    n = '.'.join(n.split('.')[:-1]) + ext
                                    output_path = os.path.join(dir_extracts, n)
                            
                            
                                    cv2.imwrite(output_path, extracted_objects)
                            

                                extracted_objects = np.array(ex, dtype = object)
                
                
                            if mask_points_values == True:
                                mask = results['masks']
                                contain_val = []
                                for a in range(mask.shape[2]):
                                    m = mask[:,:,a]
                                    mask_values = Mask(m).polygons()
                                    val = mask_values.points
                
                                    contain_val.append(val)
                                contain_val = np.array(contain_val, dtype= object)


                                results['masks'] = contain_val


                                extract_mask = extracted_objects
                                object_val = []
                            
                                
            
                            """ The mask values of each of the extracted cropped object in the image
                            is added to the dictionary containing an array of output values:
                            """ 
                            results.update({"extracted_objects":extracted_objects})
                           
                
            res.append(results)
            out.append(output)  


        return res, out   
    



    def segmentFrame(self, frame, show_bboxes = False,  extract_segmented_objects = False, 
    save_extracted_objects = False,mask_points_values = False,  output_image_name = None,
    text_thickness = 0,text_size = 0.6, box_thickness = 2, verbose = None):


        new_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        # Run detection
        if verbose is not None:
            print("Processing image...")
        results = self.model.detect([new_frame])    


        r = results[0] 
        
            
        if show_bboxes == False:

            output = display_instances(frame, r['rois'], r['masks'], r['class_ids'], self.config.class_names)

        else: 
            output = display_box_instances(frame, r['rois'], r['masks'], r['class_ids'], self.config.class_names, r['scores'],
            text_thickness = text_thickness, text_size = text_size, box_thickness = box_thickness)   

        if output_image_name is not None:
            cv2.imwrite(output_image_name, output)
            print("Processed image saved successfully in your current working directory.")   


            
        
        if extract_segmented_objects == False:
                
            if mask_points_values == True:
                mask = r['masks']
                contain_val = []
                    
                for a in range(mask.shape[2]):
                    m = mask[:,:,a]
                    mask_values = Mask(m).polygons()
                    val = mask_values.points
                    contain_val.append(val)

                contain_val = np.asarray(contain_val, dtype = object)
                r['masks'] = contain_val
             

            return r, output

        else:
            """ Code to extract and crop out each of the objects segmented in an image """


            mask = r['masks']
            m = 0
            ex = []

            if len(mask != 0):
                for a in range(mask.shape[2]):
                    
                    ori_frame = new_frame
                    img = cv2.cvtColor(ori_frame, cv2.COLOR_RGB2BGR)
                
                    
                    for b in range(img.shape[2]):
       
                        img[:,:,b] = img[:,:,b] * mask[:,:,a]
                    m+=1
                    extracted_objects = img[np.ix_(mask[:,:,a].any(1), mask[:,:,a].any(0))]
                    ex.append(extracted_objects)
                    if save_extracted_objects == True:
                        save_path = os.path.join("segmented_object" + "_" + str(m) + ".jpg")
                        cv2.imwrite(save_path, extracted_objects)
        
                    extracted_objects = np.array(ex, dtype=object)

                    if mask_points_values == True:
                
                        contain_val = []
                        for a in range(mask.shape[2]):
                            m = mask[:,:,a]
                            mask_values = Mask(m).polygons()
                            val = mask_values.points
                
                            contain_val.append(val)

                        contain_val = np.asarray(contain_val, dtype = object)
                        r['masks'] = contain_val


                        extract_mask = extracted_objects
                        object_val = []

                        for a in range(extract_mask.shape[2]):
                            m = extract_mask[:,:,a]
                            mask_values = Mask(m).polygons()
                            val = mask_values.points
                            object_val.append(val)

                        object_val = np.asarray(object_val, dtype = object)
                        extracted_objects = object_val

            
                    """ The mask values of each of the extracted cropped object in the image
                    is added to the dictionary containing an array of output values:
                    """ 
                    r.update({"extracted_objects":extracted_objects})

                return r, output

        

              

        
    def process_video(self, video_path, show_bboxes = False, extract_segmented_objects = False, save_extracted_objects = False,
    mask_points_values = False, output_video_name = None, frames_per_second = None,
    text_thickness = 0,text_size = 0.6, box_thickness = 2):
        capture = cv2.VideoCapture(video_path)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        codec = cv2.VideoWriter_fourcc(*'DIVX')
        if frames_per_second is not None:
            save_video = cv2.VideoWriter(output_video_name, codec, frames_per_second, (width, height))
        counter = 0
        start = time.time()     
           
        while True:
                
            ret, frame = capture.read()
            counter +=1

            if ret:
                seg, output =  self.segmentFrame(frame, show_bboxes=show_bboxes,extract_segmented_objects=extract_segmented_objects, 
                save_extracted_objects=save_extracted_objects,text_thickness = text_thickness,text_size = text_size, 
                box_thickness = box_thickness,mask_points_values= mask_points_values )
                    
                print("No. of frames:", counter) 
                output = cv2.resize(output, (width,height), interpolation=cv2.INTER_AREA)

                

                if output_video_name is not None:
                    save_video.write(output)

                  
                   
            else:
                break  
                 
        end = time.time() 
            
            
        print(f"Processed {counter} frames in {end-start:.1f} seconds")  
           
        capture.release()

        if output_video_name is not None:
            save_video.release()  
   
        
        return seg, output     
        
        
                  
    def process_camera(self, cam, show_bboxes = False, extract_segmented_objects = False, save_extracted_objects = False,  
    mask_points_values = False, output_video_name = None, frames_per_second = None, show_frames = None, 
    frame_name = None, text_thickness = 0,text_size = 0.6, box_thickness = 2,  verbose = None, check_fps = False):
        capture = cam
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if output_video_name is not None:
            codec = cv2.VideoWriter_fourcc(*'DIVX')
            save_video = cv2.VideoWriter(output_video_name, codec, frames_per_second, (width, height))

        counter = 0
        start = datetime.now()     

        
        while True:
                
            ret, frame = capture.read()
            if ret:
                seg, output =  self.segmentFrame(frame, show_bboxes=show_bboxes,extract_segmented_objects=extract_segmented_objects, 
                save_extracted_objects=save_extracted_objects,text_thickness = text_thickness,
                text_size = text_size, box_thickness = box_thickness,mask_points_values= mask_points_values )
                    
               
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
   
        
        return seg, output     

        



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





def display_box_instances(image, boxes, masks, class_ids, class_name, scores, text_size, box_thickness, text_thickness):
    
    n_instances = boxes.shape[0]
    colors = random_colors(n_instances)

    txt_color = (255,255,255)
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
        image = cv2.rectangle(image, (x1, y1), (x2, y2), color_rec, box_thickness)
        image = cv2.putText(image, caption, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, text_size,  txt_color, text_thickness)
        
    return image



