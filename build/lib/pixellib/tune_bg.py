import tensorflow as tf
import numpy as np
from PIL import Image
from .deeplab import Deeplab_xcep_pascal
from .semantic import obtain_segmentation
import cv2
import time
from datetime import datetime


class alter_bg():
  INPUT_TENSOR_NAME = 'ImageTensor:0'
  OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
  INPUT_SIZE = 513
  FROZEN_GRAPH_NAME = 'frozen_inference_graph'

  def __init__(self, model_type = "h5"):
    global model_file
    self.model_type = model_type
    model_file = model_type

    self.model = Deeplab_xcep_pascal()

    
  def load_pascalvoc_model(self, model_path):
    if model_file == "pb":
      self.graph = tf.Graph()

      graph_def = None

      with tf.compat.v1.gfile.GFile(model_path, 'rb')as file_handle:
        graph_def = tf.compat.v1.GraphDef.FromString(file_handle.read())

      if graph_def is None:
        raise RuntimeError('Cannot find inference graph')

      with self.graph.as_default():
        tf.graph_util.import_graph_def(graph_def, name='')

      self.sess = tf.compat.v1.Session(graph=self.graph)

    else:   
      self.model.load_weights(model_path)

  def segmentAsPascalvoc(self, image_path, process_frame = False):
    if model_file == "pb":

      if process_frame == True:
        image = image_path
      else:
        image = cv2.imread(image_path)

      h, w, n = image.shape
     
      if n > 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
      resize_ratio = 1.0 * self.INPUT_SIZE / max(w, h)
      target_size = (int(resize_ratio * w), int(resize_ratio * h))
      image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
      resized_image = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
      batch_seg_map = self.sess.run(
        self.OUTPUT_TENSOR_NAME,
        feed_dict={self.INPUT_TENSOR_NAME: [np.asarray(resized_image)]})
     
      seg_image = batch_seg_map[0]
      raw_labels = seg_image
      labels = obtain_segmentation(seg_image)
      labels = np.array(Image.fromarray(labels.astype('uint8')).resize((w, h)))
      labels = cv2.cvtColor(labels, cv2.COLOR_RGB2BGR)

      return raw_labels, labels 

    else:               
      trained_image_width=512
      mean_subtraction_value=127.5

      if process_frame == True:
        image = image_path

      else:  
        image = np.array(Image.open(image_path))     
   

      # resize to max dimension of images from training dataset
      w, h, n = image.shape

      if n > 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    

      ratio = float(trained_image_width) / np.max([w, h])
      resized_image = np.array(Image.fromarray(image.astype('uint8')).resize((int(ratio * h), int(ratio * w))))
      resized_image = (resized_image / mean_subtraction_value) -1


      # pad array to square image to match training images
      pad_x = int(trained_image_width - resized_image.shape[0])
      pad_y = int(trained_image_width - resized_image.shape[1])
      resized_image = np.pad(resized_image, ((0, pad_x), (0, pad_y), (0, 0)), mode='constant')

      #run prediction
      res = self.model.predict(np.expand_dims(resized_image, 0))
    
      labels = np.argmax(res.squeeze(), -1)
      # remove padding and resize back to original image
      if pad_x > 0:
        labels = labels[:-pad_x]
      if pad_y > 0:
        labels = labels[:, :-pad_y]

      raw_labels = labels
        
      #Apply segmentation color map
      labels = obtain_segmentation(labels)
      labels = np.array(Image.fromarray(labels.astype('uint8')).resize((h, w)))
    
    
      new_img = cv2.cvtColor(labels, cv2.COLOR_RGB2BGR)


      return raw_labels, new_img 



  ####FUNCTION TO FILTER OUT OTHER DETECTIONS AND ENSURE THE DETECTION OF A DISTINCT OBJECT####


  def target_obj(self, segment):

    self.segment = segment
    if segment == "person":
      segment = (255,255,255)
      return segment

   
    if segment == "car":
      segment = (128,128,128)  
      return segment
    
    if segment == "aeroplane":
      segment = (128,0,0) 
      return segment

    
    if segment == "bicycle": 
      segment = (0,128,0) 
      return segment

     
    if segment == "bird":
      segment = (255, 255, 255)
      return segment

     
    if segment == "boat":
      segment = (0,0,128)
      return segment

    
    if segment == "bottle":
      segment = (128,0,128)
      return segment
    
    
    if segment == "bus":
      segment = (0,128,128) 
      return segment

    
    if segment == "cat":
      segment = (64,0,0) 
      return segment

    
    if segment == "chair":  
      segment = (192,0,0)
      return segment
      
    
    if segment == "cow":
      segment = (64,128,0)
      return segment

    
    if segment == "diningtable": 
      segment = (192,128,0) 
      return segment

    if segment == "dog":
      segment = (64,0,128) 
      return segment

    
    if segment == "horse":  
      segment = (192,0,128)
      return segment

    
    if segment == "motorbike":  
      segment = (64, 128, 128) 
      return segment
    

    
    if segment == "pottedplant":
      segment = (0,64,0)
      return segment

    
    if segment == "sheep":
      segment = (128,64,0)
      return segment 
      
    
    if segment == "sofa":
      segment = (0,192,0) 
      return segment
      
    
    if segment == "train":
      segment = (128,192, 0) 
      return segment
      
    
    if segment == "monitor":
      segment = (0,64,128)  
      return segment


  #### ALTER IMAGE BACKGROUND WITH A NEW PICTURE ###

  def change_bg_img(self, f_image_path,b_image_path, output_image_name = None, verbose = None, detect = None):
    if verbose is not None:
      print("processing image......")

    seg_image = self.segmentAsPascalvoc(f_image_path)
    
    if detect is not None:
      target_class = self.target_obj(detect)
      seg_image[1][seg_image[1] != target_class] = 0
      
    ori_img = cv2.imread(f_image_path)
    
    bg_img = cv2.imread(b_image_path)
    w, h, _ = ori_img.shape
    bg_img = cv2.resize(bg_img, (h,w))

    result = np.where(seg_image[1], ori_img, bg_img)
    if output_image_name is not None:
      cv2.imwrite(output_image_name, result)

    return result


  def change_frame_bg(self, frame,b_image_path,  verbose = None, detect = None):
    if verbose is not None:
      print("processing frame......")

    seg_frame = self.segmentAsPascalvoc(frame, process_frame= True)
    
    if detect is not None:
      target_class = self.target_obj(detect)
      seg_frame[1][seg_frame[1] != target_class] = 0
      
    
    bg_img = cv2.imread(b_image_path)
    w, h, _ = frame.shape
    bg_img = cv2.resize(bg_img, (h,w))

    result = np.where(seg_frame[1], frame, bg_img)
    

    return result  

  

  ## CREATE A VIRTUAL BACKGROUND FOR A VIDEO USING AN IMAGE ##

  def change_video_bg(self, video_path, b_image_path, frames_per_second = None,output_video_name = None, detect = None):
    capture = cv2.VideoCapture(video_path)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if frames_per_second is not None:
      save_video = cv2.VideoWriter(output_video_name, cv2.VideoWriter_fourcc(*'DIVX'),frames_per_second, (width, height))
    
    counter = 0
    start = time.time() 
    
    while True:
        counter += 1
        ret, frame = capture.read()
        if ret:
            
            seg_frame = self.segmentAsPascalvoc(frame, process_frame=True)
            print("No. of frames:", counter)
            if detect is not None:
              target_class = self.target_obj(detect)
              seg_frame[1][seg_frame[1] != target_class] = 0
            w, h, _ = seg_frame[1].shape  
            img = cv2.imread(b_image_path)
            img = cv2.resize(img, (h,w))
            out = np.where(seg_frame[1], frame, img)
            

            output = cv2.resize(out, (width,height), interpolation=cv2.INTER_AREA)
            if output_video_name is not None:
                save_video.write(output)

        else:
          break

    capture.release()

    end = time.time()
    print(f"Processed {counter} frames in {end-start:.1f} seconds")
      
    if frames_per_second is not None:
        save_video.release()

    return  output  



  ## CREATE A VIRTUAL BACKGROUND FOR A CAMERA FEED USING AN IMAGE ##

  def change_camera_bg(self, cam, b_image_path, frames_per_second = None, check_fps = False,show_frames = False, 
  frame_name = None, verbose = None, output_video_name = None, detect = None):
    capture = cam
    
    if output_video_name is not None:
      width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
      height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
      save_video = cv2.VideoWriter(output_video_name, cv2.VideoWriter_fourcc(*'DIVX'),frames_per_second, (width, height))

    counter = 0
    start = datetime.now() 
     
    while True:
      ret, frame = capture.read()
      if ret:
        seg_frame = self.segmentAsPascalvoc(frame, process_frame=True)
        
        if detect is not None:
          target_class = self.target_obj(detect)
          seg_frame[1][seg_frame[1] != target_class] = 0

        w, h, _ = seg_frame[1].shape  
        img = cv2.imread(b_image_path)
        img = cv2.resize(img, (h,w))
        output = np.where(seg_frame[1], frame, img)
        counter += 1    
        
        if show_frames == True:
          if frame_name is not None:
            cv2.imshow(frame_name, output)
            if cv2.waitKey(25) & 0xFF == ord('q'):
              break 

        if output_video_name is not None:
          output = cv2.resize(output, (width,height), interpolation=cv2.INTER_AREA)
          save_video.write(output)

      elif counter == 30:

        break

    end = datetime.now()

    if check_fps == True:
      timetaken = (end-start).total_seconds()
      fps = counter/timetaken
      print(f"{fps} frames per seconds") 


    capture.release()

    if verbose is not None:
      print(f"Processed {counter} frames in {timetaken:.1f} seconds")
      
    if output_video_name is not None:
      save_video.release()

    return  output


  ##### GIVE THE BACKGROUND OF AN IMAGE A DISTINCT COLOR ######
    
  def color_bg(self, image_path, colors, output_image_name = None, verbose = None, detect = None):
    if verbose is not None:
      print("processing image......")
      
    seg_image = self.segmentAsPascalvoc(image_path)
    if detect is not None:
      target_class = self.target_obj(detect)
      seg_image[1][seg_image[1] != target_class] = 0

    ori_img = cv2.imread(image_path)
    
    obtain_img = cv2.subtract(seg_image[1], ori_img)
    out = cv2.subtract(seg_image[1], obtain_img)

    out[np.where((out == [0, 0, 0]).all(axis = 2))] = [colors]
    out = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
    result = np.where(seg_image[1], ori_img, out)


    if output_image_name is not None:
      cv2.imwrite(output_image_name, result)

    return result


   
   ##### GIVE THE BACKGROUND OF A FRAME A DISTINCT COLOR ######

  def color_frame(self, frame, colors, verbose = None, detect = None):
    if verbose is not None:
      print("processing frame....")

    seg_frame = self.segmentAsPascalvoc(frame, process_frame=True)

    if detect is not None:
      target_class = self.target_obj(detect)
      seg_frame[1][seg_frame[1] != target_class] = 0
    
    obtain_frame = cv2.subtract(seg_frame[1], frame)
    out = cv2.subtract(seg_frame[1], obtain_frame)
    out[np.where((out == [0, 0, 0]).all(axis = 2))] = [colors]
    out = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
    result = np.where(seg_frame[1], frame, out)


    return result




  ##### GIVE THE BACKGROUND OF A VIDEO A DISTINCT COLOR ######


  def color_video(self, video_path, colors, frames_per_second = None, output_video_name = None, detect = None):
    capture = cv2.VideoCapture(video_path)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if frames_per_second is not None:
      save_video = cv2.VideoWriter(output_video_name, cv2.VideoWriter_fourcc(*'DIVX'),frames_per_second, (width, height))
    counter = 0
    start = time.time() 
     
    while True:
        counter += 1
        ret, frame = capture.read()
        if ret:
          seg_frame = self.segmentAsPascalvoc(frame, process_frame=True)

          if detect is not None:
            target_class = self.target_obj(detect)
            seg_frame[1][seg_frame[1] != target_class] = 0

          print("No. of frames:", counter)
          obtain_frame = cv2.subtract(seg_frame[1], frame)
          out = cv2.subtract(seg_frame[1], obtain_frame)

          out[np.where((out == [0, 0, 0]).all(axis = 2))] = [colors]
          out = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
          out = np.where(seg_frame[1], frame, out)
          
            
          output = cv2.resize(out, (width,height), interpolation=cv2.INTER_AREA)
          if output_video_name is not None:
            save_video.write(output)

        else:

          break

    capture.release()

    end = time.time()
    print(f"Processed {counter} frames in {end-start:.1f} seconds")
      
    if frames_per_second is not None:
        save_video.release()

    return  output



  ##### GIVE THE BACKGROUND OF A CAMERA FRAME A DISTINCT COLOR ######  

  def color_camera(self, cam, colors, frames_per_second = None, check_fps = False,show_frames = False, 
  frame_name = None, verbose = None, output_video_name = None, detect = None):
    capture = cam
    
    if output_video_name is not None:
      width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
      height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
      save_video = cv2.VideoWriter(output_video_name, cv2.VideoWriter_fourcc(*'DIVX'),frames_per_second, (width, height))
    counter = 0
    start = datetime.now() 
     
    while True:
      
      ret, frame = capture.read()
      if ret:
        seg_frame = self.segmentAsPascalvoc(frame, process_frame=True)

        if detect is not None:
          target_class = self.target_obj(detect)
          seg_frame[1][seg_frame[1] != target_class] = 0

          
        obtain_frame = cv2.subtract(seg_frame[1], frame)
        out = cv2.subtract(seg_frame[1], obtain_frame)

        out[np.where((out == [0, 0, 0]).all(axis = 2))] = [colors]
        out = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
        output = np.where(seg_frame[1], frame, out)
        counter += 1
            
        
        if show_frames == True:
          if frame_name is not None:
            cv2.imshow(frame_name, output)
            if cv2.waitKey(25) & 0xFF == ord('q'):
              break 

        if output_video_name is not None:
          output = cv2.resize(output, (width,height), interpolation=cv2.INTER_AREA)
          save_video.write(output)

      elif counter == 30:

        break
    end = datetime.now()
    if check_fps == True:
      timetaken = (end-start).total_seconds()
      fps = counter/timetaken
      print(f"{fps} frames per seconds") 


    capture.release()

    if verbose is not None:
      print(f"Processed {counter} frames in {timetaken:.1f} seconds")
      
    if output_video_name is not None:
      save_video.release()

    return  output

  
  


  ##### BLUR THE BACKGROUND OF AN IMAGE #####

  def blur_bg(self, image_path,low = False, moderate = False, extreme = False, output_image_name = None, verbose = None, detect = None):
    if verbose is not None:
      print("processing image......")
      
    seg_image = self.segmentAsPascalvoc(image_path)

    if detect is not None:
      target_class = self.target_obj(detect)
      seg_image[1][seg_image[1] != target_class] = 0
    
    ori_img = cv2.imread(image_path)

    if low == True:
        blur_img = cv2.blur(ori_img, (21,21), 0)

    if moderate == True:
        blur_img = cv2.blur(ori_img, (39,39), 0)

    if extreme == True:
        blur_img = cv2.blur(ori_img, (81,81), 0)

    out = np.where(seg_image[1], ori_img, blur_img)
    
    if output_image_name is not None:
        cv2.imwrite(output_image_name, out)

    return out    



  ##### BLUR THE BACKGROUND OF A FRAME #####

  def blur_frame(self, frame,low = False, moderate = False, extreme = False, verbose = None, detect = None):
    if verbose is not None:
      print("processing frame......")
      
    seg_frame = self.segmentAsPascalvoc(frame, process_frame=True)
    if detect is not None:
      target_class = self.target_obj(detect)
      seg_frame[1][seg_frame[1] != target_class] = 0

    if low == True:
        blur_frame = cv2.blur(frame, (21,21), 0)

    if moderate == True:
        blur_frame = cv2.blur(frame, (39,39), 0)

    if extreme == True:
        blur_frame = cv2.blur(frame, (81,81), 0)

    result = np.where(seg_frame[1], frame, blur_frame)

    return result     


  ####  BLUR THE BACKGROUND OF A VIDEO #####

  def blur_video(self, video_path, low = False, moderate = False, extreme = False, frames_per_second = None,
  output_video_name = None, detect = None):
    capture = cv2.VideoCapture(video_path)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if frames_per_second is not None:
      save_video = cv2.VideoWriter(output_video_name, cv2.VideoWriter_fourcc(*'DIVX'),frames_per_second, (width, height))
    
    counter = 0
    start = time.time() 
    
    while True:
        counter += 1
        ret, frame = capture.read()
        if ret:
            
            seg_frame = self.segmentAsPascalvoc(frame, process_frame=True)
            print("No. of frames:", counter)
            if detect is not None:
              target_class = self.target_obj(detect)
              seg_frame[1][seg_frame[1] != target_class] = 0

            if low == True:
                blur_frame = cv2.blur(frame, (21,21), 0)

            if moderate == True:
                blur_frame = cv2.blur(frame, (39,39), 0)

            if extreme == True:
                blur_frame = cv2.blur(frame, (81,81), 0)

            out = np.where(seg_frame[1], frame, blur_frame)
            

            output = cv2.resize(out, (width,height), interpolation=cv2.INTER_AREA)
            if output_video_name is not None:
                save_video.write(output)

        else:
          break

    capture.release()

    end = time.time()
    print(f"Processed {counter} frames in {end-start:.1f} seconds")
      
    if frames_per_second is not None:
        save_video.release()

    return  output  


   ##### BLUR THE BACKGROUND OF A CAMERA FRAME ######  

  def blur_camera(self, cam, low = False, moderate = False, extreme = False, frames_per_second = None,
   check_fps = False,show_frames = False, frame_name = None, verbose = None, output_video_name = None, detect = None):
    capture = cam

    if output_video_name is not None:
      width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
      height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
      save_video = cv2.VideoWriter(output_video_name, cv2.VideoWriter_fourcc(*'DIVX'),frames_per_second, (width, height))
      
    counter = 0
    start = datetime.now() 
     
    while True:
        
      ret, frame = capture.read()
      if ret:

        seg_frame = self.segmentAsPascalvoc(frame, process_frame=True)
        if detect is not None:
          target_class = self.target_obj(detect)
          seg_frame[1][seg_frame[1] != target_class] = 0
            
    
        if low == True:
          blur_frame = cv2.blur(frame, (21,21), 0)

        if moderate == True:
          blur_frame = cv2.blur(frame, (39,39), 0)

        if extreme == True:
          blur_frame = cv2.blur(frame, (81,81), 0)

        output = np.where(seg_frame[1], frame, blur_frame)  
        counter += 1
          
        if show_frames == True:
          if frame_name is not None:
            cv2.imshow(frame_name, output)
            if cv2.waitKey(25) & 0xFF == ord('q'):
              break 

        if output_video_name is not None:
          output = cv2.resize(output, (width,height), interpolation=cv2.INTER_AREA)
          save_video.write(output)


      elif couter == 30:
        break

    end = datetime.now()   
    if check_fps == True:
      timetaken = (end-start).total_seconds()
      fps = counter/timetaken
      print(f"{fps} frames per seconds") 

    capture.release()

    
    if verbose is not None:
      print(f"Processed {counter} frames in {timetaken:.1f} seconds")
      
    if output_video_name is not None:
      save_video.release()

    return  output


   ### GRAYSCALE THE BACKGROUND OF AN IMAGE ###

  def gray_bg(self, image_path, output_image_name = None, verbose = None, detect = None):
    if verbose is not None:
      print("processing image......")
      
    seg_image = self.segmentAsPascalvoc(image_path)

    if detect is not None:
      target_class = self.target_obj(detect)
      seg_image[1][seg_image[1] != target_class] = 0
    
    ori_img = cv2.imread(image_path)
    
    gray_img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2GRAY)
    gray_img = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)
    result = np.where(seg_image[1], ori_img, gray_img)
    
    if output_image_name is not None:
        cv2.imwrite(output_image_name, result)

    return result



  def gray_frame(self, frame, verbose = None, detect = None):
    if verbose is not None:
      print("processing frame......")
      
    seg_frame = self.segmentAsPascalvoc(frame, process_frame=True)
    if detect is not None:
      target_class = self.target_obj(detect)
      seg_frame[1][seg_frame[1] != target_class] = 0
    
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)
    result = np.where(seg_frame[1], frame, gray_frame)

    return result  


   ### GRAYSCALE THE BACKGROUND OF A VIDEO ###

  def gray_video(self, video_path, frames_per_second = None, output_video_name = None, detect = None):
    capture = cv2.VideoCapture(video_path)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if frames_per_second is not None:
      save_video = cv2.VideoWriter(output_video_name, cv2.VideoWriter_fourcc(*'DIVX'),frames_per_second, (width, height))
    
    counter = 0
    start = time.time() 
    
    while True:
        counter += 1
        ret, frame = capture.read()
        if ret:
            seg_frame = self.segmentAsPascalvoc(frame, process_frame=True)
            if detect is not None:
              target_class = self.target_obj(detect)
              seg_frame[1][seg_frame[1] != target_class] = 0

            print("No. of frames:", counter)

            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)
            out = np.where(seg_frame[1], frame, gray_frame)

            output = cv2.resize(out, (width,height), interpolation=cv2.INTER_AREA)
            if output_video_name is not None:
                save_video.write(output)

        else:
          break

    capture.release()

    end = time.time()
    print(f"Processed {counter} frames in {end-start:.1f} seconds")
      
    if frames_per_second is not None:
        save_video.release()

    return  output
      

  ### GRAYSCALE THE BACKGROUND OF A CAMERA FEED ###

  def gray_camera(self, cam, frames_per_second = None, check_fps = False,show_frames = False, 
  frame_name = None, verbose = None, output_video_name = None, detect = None):

    capture = cam
    
    if output_video_name is not None:
      width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
      height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
      save_video = cv2.VideoWriter(output_video_name, cv2.VideoWriter_fourcc(*'DIVX'),frames_per_second, (width, height))
    counter = 0
    start = datetime.now() 
     
    while True:
      
      ret, frame = capture.read()
      if ret:
        seg_frame = self.segmentAsPascalvoc(frame, process_frame=True)
        if detect is not None:
          target_class = self.target_obj(detect)
          seg_frame[1][seg_frame[1] != target_class] = 0

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)
        output = np.where(seg_frame[1], frame, gray_frame)
        counter += 1
          
        
        if show_frames == True:
          if frame_name is not None:
            cv2.imshow(frame_name, output)
            if cv2.waitKey(25) & 0xFF == ord('q'):
              break 

        if output_video_name is not None:
          output = cv2.resize(output, (width,height), interpolation=cv2.INTER_AREA)
          save_video.write(output)

      elif counter == 30:

        break

    end = datetime.now()
    if check_fps == True:
      timetaken = (end-start).total_seconds()
      fps = counter/timetaken
      print(f"{fps} frames per seconds") 
      

    capture.release()

    
    if verbose is not None:
      print(f"Processed {counter} frames in {timetaken:.1f} seconds")
      
    if output_video_name is not None:
      save_video.release()

    return  output
    