import tensorflow as tf
import numpy as np
from PIL import Image
from .deeplab import Deeplab_xcep_pascal
from .deeplab import Deeplab_xcep_ade20k
import cv2
import time


class semantic_segmentation():

  def __init__(self):

    self.model = Deeplab_xcep_pascal()

    self.model2 = Deeplab_xcep_ade20k()
      
  def load_pascalvoc_model(self, model_path):
    self.model.load_weights(model_path)
  
  def load_ade20k_model(self, model_path):
    self.model2.load_weights(model_path)   

  def segmentAsPascalvoc(self, image_path, output_image_name=None,overlay=False, verbose = None):            
    trained_image_width=512
    mean_subtraction_value=127.5
    image = np.array(Image.open(image_path))     
   

    # resize to max dimension of images from training dataset
    w, h, n = image.shape

    if n > 3:
      image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    
    image_overlay = image.copy()

    ratio = float(trained_image_width) / np.max([w, h])
    resized_image = np.array(Image.fromarray(image.astype('uint8')).resize((int(ratio * h), int(ratio * w))))
    resized_image = (resized_image / mean_subtraction_value) -1


    # pad array to square image to match training images
    pad_x = int(trained_image_width - resized_image.shape[0])
    pad_y = int(trained_image_width - resized_image.shape[1])
    resized_image = np.pad(resized_image, ((0, pad_x), (0, pad_y), (0, 0)), mode='constant')

    if verbose is not None:
      print("Processing image....")

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
    labels = labelP_to_color_image(labels)   
    labels = np.array(Image.fromarray(labels.astype('uint8')).resize((h, w)))
    
    
    new_img = cv2.cvtColor(labels, cv2.COLOR_RGB2BGR)

    if overlay == True:
        alpha = 0.7
        cv2.addWeighted(new_img, alpha, image_overlay, 1 - alpha,0, image_overlay)

        if output_image_name is not None:
          cv2.imwrite(output_image_name, image_overlay)
          print("Processed Image saved successfully in your current working directory.")

        return raw_labels, image_overlay

        
    else:  
        if output_image_name is not None:
  
          cv2.imwrite(output_image_name, new_img)

          print("Processed Image saved successfuly in your current working directory.")

        return raw_labels, new_img 

        
  def segmentAsAde20k(self, image_path, output_image_name=None,overlay=False, verbose = None):            
    trained_image_width=512
    mean_subtraction_value=127.5
    image = np.array(Image.open(image_path))     
    
    # resize to max dimension of images from training dataset
    w, h, n = image.shape

    if n > 3:
      image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    
    image_overlay = image.copy()

    ratio = float(trained_image_width) / np.max([w, h])
    resized_image = np.array(Image.fromarray(image.astype('uint8')).resize((int(ratio * h), int(ratio * w))))
    resized_image = (resized_image / mean_subtraction_value) -1


    # pad array to square image to match training images
    pad_x = int(trained_image_width - resized_image.shape[0])
    pad_y = int(trained_image_width - resized_image.shape[1])
    resized_image = np.pad(resized_image, ((0, pad_x), (0, pad_y), (0, 0)), mode='constant')

    if verbose is not None:
      print("Processing image....")
    #run prediction
    res = self.model2.predict(np.expand_dims(resized_image, 0))
    
    labels = np.argmax(res.squeeze(), -1)
    # remove padding and resize back to original image
    if pad_x > 0:
        labels = labels[:-pad_x]
    if pad_y > 0:
        labels = labels[:, :-pad_y]

    raw_labels = labels    
        
    #Apply segmentation color map
    labels = labelAde20k_to_color_image(labels)   
    labels = np.array(Image.fromarray(labels.astype('uint8')).resize((h, w)))
    
    
    new_img = cv2.cvtColor(labels, cv2.COLOR_RGB2BGR)

    if overlay == True:
        alpha = 0.7
        cv2.addWeighted(new_img, alpha, image_overlay, 1 - alpha,0, image_overlay)

        if output_image_name is not None:
          cv2.imwrite(output_image_name, image_overlay)
          print("Processed Image saved successfully in your current working directory.")

        return raw_labels, image_overlay 

        
    else:  
        if output_image_name is not None:
  
          cv2.imwrite(output_image_name, new_img)

          print("Processed Image saved successfuly in your current working directory.")

        return raw_labels, new_img 

  def segmentFrameAsPascalvoc(self, frame, output_image_name=None,overlay=False, verbose = None):            
    trained_frame_width=512
    mean_subtraction_value=127.5
       
    frame_overlay = frame.copy()

    # resize to max dimension of images from training dataset
    w, h, _ = frame.shape
    ratio = float(trained_frame_width) / np.max([w, h])
    resized_frame = np.array(Image.fromarray(frame.astype('uint8')).resize((int(ratio * h), int(ratio * w))))
    resized_frame = (resized_frame / mean_subtraction_value) -1


    # pad array to square image to match training images
    pad_x = int(trained_frame_width - resized_frame.shape[0])
    pad_y = int(trained_frame_width - resized_frame.shape[1])
    resized_frame = np.pad(resized_frame, ((0, pad_x), (0, pad_y), (0, 0)), mode='constant')

    if verbose is not None:
      print("Processing frame....")

    #run prediction
    res = self.model.predict(np.expand_dims(resized_frame, 0))
    
    labels = np.argmax(res.squeeze(), -1)
    # remove padding and resize back to original image
    if pad_x > 0:
        labels = labels[:-pad_x]
    if pad_y > 0:
        labels = labels[:, :-pad_y]
        
    raw_labels = labels    
    #Apply segmentation color map
    labels = labelP_to_color_image(labels)   
    labels = np.array(Image.fromarray(labels.astype('uint8')).resize((h, w)))
    
    
    new_frame = cv2.cvtColor(labels, cv2.COLOR_RGB2BGR)

    if overlay == True:
        alpha = 0.7
        cv2.addWeighted(new_frame, alpha, frame_overlay, 1 - alpha,0, frame_overlay)

        if output_image_name is not None:
          cv2.imwrite(output_image_name, frame_overlay)
          print("Processed Image saved successfully in your current working directory.")

        return  raw_labels, frame_overlay

        
    else:  
        if output_image_name is not None:
  
          cv2.imwrite(output_image_name, new_frame)

          print("Processed Image saved successfuly in your current working directory.")

        return raw_labels, new_frame



  def segmentFrameAsAde20k(self, frame, output_image_name=None,overlay=False, verbose = None):            
    trained_frame_width=512
    mean_subtraction_value=127.5     
    frame_overlay = frame.copy()

    # resize to max dimension of images from training dataset
    w, h, _ = frame.shape
    ratio = float(trained_frame_width) / np.max([w, h])
    resized_frame = np.array(Image.fromarray(frame.astype('uint8')).resize((int(ratio * h), int(ratio * w))))
    resized_frame = (resized_frame / mean_subtraction_value) -1


    # pad array to square image to match training images
    pad_x = int(trained_frame_width - resized_frame.shape[0])
    pad_y = int(trained_frame_width - resized_frame.shape[1])
    resized_frame = np.pad(resized_frame, ((0, pad_x), (0, pad_y), (0, 0)), mode='constant')

    if verbose is not None:
      print("Processing frame....")

    #run prediction
    res = self.model2.predict(np.expand_dims(resized_frame, 0))
    
    labels = np.argmax(res.squeeze(), -1)
    # remove padding and resize back to original image
    if pad_x > 0:
        labels = labels[:-pad_x]
    if pad_y > 0:
        labels = labels[:, :-pad_y]

    raw_labels = labels    
    #Apply segmentation color map
    labels = labelAde20k_to_color_image(labels)   
    labels = np.array(Image.fromarray(labels.astype('uint8')).resize((h, w)))
    
    
    new_frame = cv2.cvtColor(labels, cv2.COLOR_RGB2BGR)

    if overlay == True:
        alpha = 0.7
        cv2.addWeighted(new_frame, alpha, frame_overlay, 1 - alpha,0, frame_overlay)

        if output_image_name is not None:
          cv2.imwrite(output_image_name, frame_overlay)
          print("Processed Image saved successfully in your current working directory.")

        return raw_labels, frame_overlay  

        
    else:  
        if output_image_name is not None:
  
          cv2.imwrite(output_image_name, new_frame)

          print("Processed Image saved successfuly in your current working directory.")

        return raw_labels, new_frame      




  
  def process_video_pascalvoc(self, video_path, overlay = False, frames_per_second = None, output_video_name = None):
    capture = cv2.VideoCapture(video_path)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if frames_per_second is not None:
      save_video = cv2.VideoWriter(output_video_name, cv2.VideoWriter_fourcc(*'DIVX'),frames_per_second, (width, height))
    
    counter = 0
    start = time.time() 


    if overlay == True:
      while True:
        counter += 1
        ret, frame = capture.read()
        if ret:
          frame_copy = frame.copy()
          trained_frame_width = 512
          mean_subtraction_value = 127.5

          # resize to max dimension of images from training dataset
          w, h, _ = frame.shape
          ratio = float(trained_frame_width) / np.max([w, h])
          resized_frame = np.array(Image.fromarray(frame.astype('uint8')).resize((int(ratio * h), int(ratio * w))))
          resized_frame = (resized_frame / mean_subtraction_value) -1


          # pad array to square image to match training images
          pad_x = int(trained_frame_width - resized_frame.shape[0])
          pad_y = int(trained_frame_width - resized_frame.shape[1])
          resized_frame = np.pad(resized_frame, ((0, pad_x), (0, pad_y), (0, 0)), mode='constant')

          pred = self.model.predict(np.expand_dims(resized_frame, axis = 0))
          
          print("No.of frames:", counter)
            
          labels = np.argmax(pred.squeeze(), -1)

          if pad_x > 0:
            labels = labels[:-pad_x]
          if pad_y > 0:
            labels = labels[:, :-pad_y]

          raw_labels = labels  
          segmap = labelP_to_color_image(labels)
          labels = np.array(Image.fromarray(segmap.astype('uint8')).resize((h, w)))
          new_segmap = cv2.cvtColor(labels, cv2.COLOR_BGR2RGB)
            
          alpha = 0.7
          cv2.addWeighted(new_segmap, alpha, frame_copy, 1-alpha,0, frame_copy)
          output = cv2.resize(frame_copy, (width,height), interpolation=cv2.INTER_AREA)
          if output_video_name is not None:
            save_video.write(output)
        else:
          break   

    
      end = time.time()
      print(f"Processed {counter} frames in {end-start:.1f} seconds")
      capture.release()
      if frames_per_second is not None:
        save_video.release()
      return  raw_labels, output

    else:
      while True:
        counter += 1
        ret, frame = capture.read()
        if ret:
          trained_frame_width = 512
          mean_subtraction_value = 127.5

          # resize to max dimension of images from training dataset
          w, h, _ = frame.shape
          ratio = float(trained_frame_width) / np.max([w, h])
          resized_frame = np.array(Image.fromarray(frame.astype('uint8')).resize((int(ratio * h), int(ratio * w))))
          resized_frame = (resized_frame / mean_subtraction_value) -1


          # pad array to square image to match training images
          pad_x = int(trained_frame_width - resized_frame.shape[0])
          pad_y = int(trained_frame_width - resized_frame.shape[1])
          resized_frame = np.pad(resized_frame, ((0, pad_x), (0, pad_y), (0, 0)), mode='constant')

          pred = self.model.predict(np.expand_dims(resized_frame, axis = 0))
          print("No.of frames:", counter)
            
          labels = np.argmax(pred.squeeze(), -1)

          if pad_x > 0:
            labels = labels[:-pad_x]
          if pad_y > 0:
            labels = labels[:, :-pad_y]

          raw_labels = labels  
          segmap = labelP_to_color_image(labels)
          labels = np.array(Image.fromarray(segmap.astype('uint8')).resize((h, w)))
          new_segmap = cv2.cvtColor(labels, cv2.COLOR_BGR2RGB)
            
          output = cv2.resize(new_segmap, (width,height), interpolation=cv2.INTER_AREA)
          if output_video_name is not None:
            save_video.write(output)

        else:
          break

      capture.release()

      end = time.time()
      print(f"Processed {counter} frames in {end-start:.1f} seconds")
      
      if frames_per_second is not None:
        save_video.release()

      return  raw_labels,  output

  

  def process_camera_pascalvoc(self, cam, overlay = False,  check_fps = False, frames_per_second = None, output_video_name = None, show_frames = False, frame_name = None, verbose = None):
    capture = cam
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if frames_per_second is not None:
      save_video = cv2.VideoWriter(output_video_name, cv2.VideoWriter_fourcc(*'DIVX'), frames_per_second, (width, height))
    
    counter = 0
    start = time.time() 
    

    if overlay == True:
      while True:
        counter += 1
        ret, frame = capture.read()
        if ret:
          frame_copy = frame.copy()
          trained_frame_width = 512
          mean_subtraction_value = 127.5

          # resize to max dimension of images from training dataset
          w, h, _ = frame.shape
          ratio = float(trained_frame_width) / np.max([w, h])
          resized_frame = np.array(Image.fromarray(frame.astype('uint8')).resize((int(ratio * h), int(ratio * w))))
          resized_frame = (resized_frame / mean_subtraction_value) -1


          # pad array to square image to match training images
          pad_x = int(trained_frame_width - resized_frame.shape[0])
          pad_y = int(trained_frame_width - resized_frame.shape[1])
          resized_frame = np.pad(resized_frame, ((0, pad_x), (0, pad_y), (0, 0)), mode='constant')

          pred = self.model.predict(np.expand_dims(resized_frame, axis = 0))
          if verbose is not None:
            print("No.of frames:", counter)
            
          labels = np.argmax(pred.squeeze(), -1)

          if pad_x > 0:
            labels = labels[:-pad_x]
          if pad_y > 0:
            labels = labels[:, :-pad_y]

          raw_labels = labels  
          segmap = labelP_to_color_image(labels)
          labels = np.array(Image.fromarray(segmap.astype('uint8')).resize((h, w)))
          new_segmap = cv2.cvtColor(labels, cv2.COLOR_BGR2RGB)
            
          alpha = 0.7
          cv2.addWeighted(new_segmap, alpha, frame_copy, 1-alpha,0, frame_copy)
          output = cv2.resize(frame_copy, (width,height), interpolation=cv2.INTER_AREA)

          if show_frames == True:
            if frame_name is not None:
              cv2.imshow(frame_name, output)
              if cv2.waitKey(25) & 0xFF == ord('q'):
                break 

          if output_video_name is not None:
            save_video.write(output)

         

        else:
          break 

      if check_fps == True:
        out = capture.get(cv2.CAP_PROP_FPS)
        print(f"{out} frames per seconds")

      capture.release()

      end = time.time()
      if verbose is not None:
        print(f"Processed {counter} frames in {end-start:.1f} seconds")
      
      if frames_per_second is not None:
        save_video.release()

      return  raw_labels, output

    else:
      while True:
        counter += 1
        ret, frame = capture.read()
        if ret:
          trained_frame_width = 512
          mean_subtraction_value = 127.5

          # resize to max dimension of images from training dataset
          w, h, _ = frame.shape
          ratio = float(trained_frame_width) / np.max([w, h])
          resized_frame = np.array(Image.fromarray(frame.astype('uint8')).resize((int(ratio * h), int(ratio * w))))
          resized_frame = (resized_frame / mean_subtraction_value) -1


          # pad array to square image to match training images
          pad_x = int(trained_frame_width - resized_frame.shape[0])
          pad_y = int(trained_frame_width - resized_frame.shape[1])
          resized_frame = np.pad(resized_frame, ((0, pad_x), (0, pad_y), (0, 0)), mode='constant')

          pred = self.model.predict(np.expand_dims(resized_frame, axis = 0))
          if verbose is not None:
            print("No.of frames:", counter)
            
          labels = np.argmax(pred.squeeze(), -1)

          if pad_x > 0:
            labels = labels[:-pad_x]
          if pad_y > 0:
            labels = labels[:, :-pad_y]

          raw_labels = labels  
          segmap = labelP_to_color_image(labels)
          labels = np.array(Image.fromarray(segmap.astype('uint8')).resize((h, w)))
          new_segmap = cv2.cvtColor(labels, cv2.COLOR_BGR2RGB)
            

          output = cv2.resize(new_segmap, (width,height), interpolation=cv2.INTER_AREA)

          if show_frames == True:
            if frame_name is not None:
              cv2.imshow(frame_name, output)
              if cv2.waitKey(25) & 0xFF == ord('q'):
                break 

          if output_video_name is not None:
            save_video.write(output)

        else:
          break
      
      if check_fps == True:
        out = capture.get(cv2.CAP_PROP_FPS)
        print("Frame per seconds:", out)

      capture.release()  
        
      end = time.time()
      if verbose is not None:
        print(f"Processed {counter} frames in {end-start:.1f} seconds")
     

      if frames_per_second is not None:
        save_video.release()

      return raw_labels, output 

  


  def process_video_ade20k(self, video_path, overlay = False, frames_per_second = None, output_video_name = None):
    capture = cv2.VideoCapture(video_path)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if frames_per_second is not None:
      save_video = cv2.VideoWriter(output_video_name, cv2.VideoWriter_fourcc(*'DIVX'), frames_per_second, (width, height))
    
    counter = 0
    start = time.time() 
    

    if overlay == True:
      while True:
        counter += 1
        ret, frame = capture.read()
        if ret:
          frame_copy = frame.copy()
          trained_frame_width = 512
          mean_subtraction_value = 127.5

          # resize to max dimension of images from training dataset
          w, h, _ = frame.shape
          ratio = float(trained_frame_width) / np.max([w, h])
          resized_frame = np.array(Image.fromarray(frame.astype('uint8')).resize((int(ratio * h), int(ratio * w))))
          resized_frame = (resized_frame / mean_subtraction_value) -1


          # pad array to square image to match training images
          pad_x = int(trained_frame_width - resized_frame.shape[0])
          pad_y = int(trained_frame_width - resized_frame.shape[1])
          resized_frame = np.pad(resized_frame, ((0, pad_x), (0, pad_y), (0, 0)), mode='constant')

          pred = self.model2.predict(np.expand_dims(resized_frame, axis = 0))
          print("No.of frames:", counter)
            
          labels = np.argmax(pred.squeeze(), -1)

          if pad_x > 0:
            labels = labels[:-pad_x]
          if pad_y > 0:
            labels = labels[:, :-pad_y]
            
          raw_labels = labels  
          segmap = labelAde20k_to_color_image(labels)
          labels = np.array(Image.fromarray(segmap.astype('uint8')).resize((h, w)))
          new_segmap = cv2.cvtColor(labels, cv2.COLOR_BGR2RGB)
            
          alpha = 0.7
          cv2.addWeighted(new_segmap, alpha, frame_copy, 1-alpha,0, frame_copy)
          output = cv2.resize(frame_copy, (width,height), interpolation=cv2.INTER_AREA)

          if output_video_name is not None:
            save_video.write(output)

        else:
          break  
             
      capture.release()

      end = time.time()
      print(f"Processed {counter} frames in {end-start:.1f} seconds")

      
      if frames_per_second is not None:
        save_video.release()

      return raw_labels, output

    else:
      while True:
        counter += 1
        ret, frame = capture.read()
        if ret:
          frame_copy = frame.copy()
          trained_frame_width = 512
          mean_subtraction_value = 127.5

          # resize to max dimension of images from training dataset
          w, h, _ = frame.shape
          ratio = float(trained_frame_width) / np.max([w, h])
          resized_frame = np.array(Image.fromarray(frame.astype('uint8')).resize((int(ratio * h), int(ratio * w))))
          resized_frame = (resized_frame / mean_subtraction_value) -1


          # pad array to square image to match training images
          pad_x = int(trained_frame_width - resized_frame.shape[0])
          pad_y = int(trained_frame_width - resized_frame.shape[1])
          resized_frame = np.pad(resized_frame, ((0, pad_x), (0, pad_y), (0, 0)), mode='constant')

          pred = self.model2.predict(np.expand_dims(resized_frame, axis = 0))
          print("No.of frames:", counter)
            
          labels = np.argmax(pred.squeeze(), -1)

          if pad_x > 0:
            labels = labels[:-pad_x]
          if pad_y > 0:
            labels = labels[:, :-pad_y]


          raw_labels = labels  
          segmap = labelAde20k_to_color_image(labels)
          labels = np.array(Image.fromarray(segmap.astype('uint8')).resize((h, w)))
          new_segmap = cv2.cvtColor(labels, cv2.COLOR_BGR2RGB)
            
          output = cv2.resize(new_segmap, (width,height), interpolation=cv2.INTER_AREA)

          if output_video_name is not None:
            save_video.write(output)



        else:
          break


      capture.release()  

      end = time.time()
      print(f"Processed {counter} frames in {end-start:.1f} seconds")

      
      if frames_per_second is not None:
        save_video.release()

      return raw_labels,  output    


  




  def process_camera_ade20k(self, cam, overlay = False,  check_fps = False, frames_per_second = None, output_video_name = None, show_frames = False, frame_name = None, verbose = None):
    capture = cam
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if frames_per_second is not None:
      save_video = cv2.VideoWriter(output_video_name, cv2.VideoWriter_fourcc(*'DIVX'), frames_per_second, (width, height))
    
    counter = 0
    start = time.time() 
    

    if overlay == True:
      while True:
        counter += 1
        ret, frame = capture.read()
        if ret:
          frame_copy = frame.copy()
          trained_frame_width = 512
          mean_subtraction_value = 127.5

          # resize to max dimension of images from training dataset
          w, h, _ = frame.shape
          ratio = float(trained_frame_width) / np.max([w, h])
          resized_frame = np.array(Image.fromarray(frame.astype('uint8')).resize((int(ratio * h), int(ratio * w))))
          resized_frame = (resized_frame / mean_subtraction_value) -1


          # pad array to square image to match training images
          pad_x = int(trained_frame_width - resized_frame.shape[0])
          pad_y = int(trained_frame_width - resized_frame.shape[1])
          resized_frame = np.pad(resized_frame, ((0, pad_x), (0, pad_y), (0, 0)), mode='constant')

          pred = self.model2.predict(np.expand_dims(resized_frame, axis = 0))
          if verbose is not None:
            print("No.of frames:", counter)
            
          labels = np.argmax(pred.squeeze(), -1)

          if pad_x > 0:
            labels = labels[:-pad_x]
          if pad_y > 0:
            labels = labels[:, :-pad_y]

          raw_labels = labels  
          segmap = labelAde20k_to_color_image(labels)
          labels = np.array(Image.fromarray(segmap.astype('uint8')).resize((h, w)))
          new_segmap = cv2.cvtColor(labels, cv2.COLOR_BGR2RGB)
            
          alpha = 0.7
          cv2.addWeighted(new_segmap, alpha, frame_copy, 1-alpha,0, frame_copy)
          output = cv2.resize(frame_copy, (width,height), interpolation=cv2.INTER_AREA)

          if show_frames == True:
            if frame_name is not None:
              cv2.imshow(frame_name, output)
              if cv2.waitKey(25) & 0xFF == ord('q'):
                break 

          if output_video_name is not None:
            save_video.write(output)

        else:
          break 

      if check_fps == True:
        out = capture.get(cv2.CAP_PROP_FPS)
        print(f"{out} frames per seconds")    
             
      capture.release()

      end = time.time()
      if verbose is not None:
        print(f"Processed {counter} frames in {end-start:.1f} seconds")

      
      if frames_per_second is not None:
        save_video.release()

      return raw_labels, output

    else:
      while True:
        counter += 1
        ret, frame = capture.read()
        if ret:
          frame_copy = frame.copy()
          trained_frame_width = 512
          mean_subtraction_value = 127.5

          # resize to max dimension of images from training dataset
          w, h, _ = frame.shape
          ratio = float(trained_frame_width) / np.max([w, h])
          resized_frame = np.array(Image.fromarray(frame.astype('uint8')).resize((int(ratio * h), int(ratio * w))))
          resized_frame = (resized_frame / mean_subtraction_value) -1


          # pad array to square image to match training images
          pad_x = int(trained_frame_width - resized_frame.shape[0])
          pad_y = int(trained_frame_width - resized_frame.shape[1])
          resized_frame = np.pad(resized_frame, ((0, pad_x), (0, pad_y), (0, 0)), mode='constant')

          pred = self.model2.predict(np.expand_dims(resized_frame, axis = 0))
          if verbose is not None:
            print("No.of frames:", counter)
            
          labels = np.argmax(pred.squeeze(), -1)

          if pad_x > 0:
            labels = labels[:-pad_x]
          if pad_y > 0:
            labels = labels[:, :-pad_y]

          raw_labels = labels  
          segmap = labelAde20k_to_color_image(labels)
          labels = np.array(Image.fromarray(segmap.astype('uint8')).resize((h, w)))
          new_segmap = cv2.cvtColor(labels, cv2.COLOR_BGR2RGB)
            
          output = cv2.resize(new_segmap, (width,height), interpolation=cv2.INTER_AREA)

          if show_frames == True:
            if frame_name is not None:
              cv2.imshow(frame_name, output)
              if cv2.waitKey(25) & 0xFF == ord('q'):
                break 

          if output_video_name is not None:
            save_video.write(output)



        else:
          break

      if check_fps == True:
        out = capture.get(cv2.CAP_PROP_FPS)
        print("Frame per seconds:", out)   

      capture.release()  

      end = time.time()
      if verbose is not None:
        print(f"Processed {counter} frames in {end-start:.1f} seconds")

      
      if frames_per_second is not None:
          save_video.release()

      return raw_labels,  output    


  
##Create Pascalvoc colormap format ##
    
def create_pascal_label_colormap():
  """Creates a label colormap used in PASCAL VOC segmentation benchmark.

  Returns:
    A Colormap for visualizing segmentation results.
  """
  colormap = np.zeros((256, 3), dtype = int)
  ind = np.arange(256, dtype=int)

  for shift in reversed(range(8)):
    for channel in range(3):
      colormap[:, channel] |= ((ind >> channel) & 1) << shift
    ind >>= 3

  return colormap


#Obtain the results of segmentation#
def label_pascal():
    return np.asarray([
    [0, 0, 0],
    [128, 0, 0], 
    [0, 128, 0], 
    [255, 255, 255], 
    [0, 0, 128], 
    [128, 0, 128],
    [0, 128, 128], 
    [128, 128, 128], 
    [64, 0, 0], 
    [192, 0, 0], 
    [64, 128, 0],
    [192, 128, 0], 
    [64, 0, 128], 
    [192, 0, 128], 
    [64, 128, 128], 
    [255, 255, 255],
    [0, 64, 0], 
    [128, 64, 0], 
    [0, 192, 0], 
    [128, 192, 0], 
    [0, 64, 128],
])  

def obtain_segmentation(image, nc = 21):
    colors = label_pascal()
    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)
    
    for a in range(0,nc):
        index = image == a
        r[index] = colors[a, 0]
        g[index] = colors[a, 1]
        b[index] = colors[a, 2]
        rgb = np.stack([r,g,b], axis = 2)  

    return rgb  

# Assign colors to objects #

def labelP_to_color_image(label):
  """Adds color defined by the dataset colormap to the label.

  Args:
    label: A 2D array with integer type, storing the segmentation label.

  Returns:
    result: A 2D array with floating type. The element of the array
      is the color indexed by the corresponding element in the input label
      to the PASCAL color map.

  Raises:
    ValueError: If label is not of rank 2 or its value is larger than color
      map maximum entry.
  """
  if label.ndim != 2:
    raise ValueError('Expect 2-D input label')

  colormap = create_pascal_label_colormap()

  if np.max(label) >= len(colormap):
    raise ValueError('label value too large.')

  return colormap[label]   



##Create Ade20k colormap format ##

def create_ade20k_label_colormap():
  """Creates a label colormap used in ADE20K segmentation benchmark.
  Returns:
    A colormap for visualizing segmentation results.
  """
  return np.asarray([
      [0, 0, 0],
      [120, 120, 120],
      [180, 120, 120],
      [6, 230, 230],
      [80, 50, 50],
      [4, 200, 3],
      [120, 120, 80],
      [140, 140, 140],
      [204, 5, 255],
      [230, 230, 230],
      [4, 250, 7],
      [224, 5, 255],
      [235, 255, 7],
      [150, 5, 61],
      [120, 120, 70],
      [8, 255, 51],
      [255, 6, 82],
      [143, 255, 140],
      [204, 255, 4],
      [255, 51, 7],
      [204, 70, 3],
      [0, 102, 200],
      [61, 230, 250],
      [255, 6, 51],
      [11, 102, 255],
      [255, 7, 71],
      [255, 9, 224],
      [9, 7, 230],
      [220, 220, 220],
      [255, 9, 92],
      [112, 9, 255],
      [8, 255, 214],
      [7, 255, 224],
      [255, 184, 6],
      [10, 255, 71],
      [255, 41, 10],
      [7, 255, 255],
      [224, 255, 8],
      [102, 8, 255],
      [255, 61, 6],
      [255, 194, 7],
      [255, 122, 8],
      [0, 255, 20],
      [255, 8, 41],
      [255, 5, 153],
      [6, 51, 255],
      [235, 12, 255],
      [160, 150, 20],
      [0, 163, 255],
      [140, 140, 140],
      [250, 10, 15],
      [20, 255, 0],
      [31, 255, 0],
      [255, 31, 0],
      [255, 224, 0],
      [153, 255, 0],
      [0, 0, 255],
      [255, 71, 0],
      [0, 235, 255],
      [0, 173, 255],
      [31, 0, 255],
      [11, 200, 200],
      [255, 82, 0],
      [0, 255, 245],
      [0, 61, 255],
      [0, 255, 112],
      [0, 255, 133],
      [255, 0, 0],
      [255, 163, 0],
      [255, 102, 0],
      [194, 255, 0],
      [0, 143, 255],
      [51, 255, 0],
      [0, 82, 255],
      [0, 255, 41],
      [0, 255, 173],
      [10, 0, 255],
      [173, 255, 0],
      [0, 255, 153],
      [255, 92, 0],
      [255, 0, 255],
      [255, 0, 245],
      [255, 0, 102],
      [255, 173, 0],
      [255, 0, 20],
      [255, 184, 184],
      [0, 31, 255],
      [0, 255, 61],
      [0, 71, 255],
      [255, 0, 204],
      [0, 255, 194],
      [0, 255, 82],
      [0, 10, 255],
      [0, 112, 255],
      [51, 0, 255],
      [0, 194, 255],
      [0, 122, 255],
      [0, 255, 163],
      [255, 153, 0],
      [0, 255, 10],
      [255, 112, 0],
      [143, 255, 0],
      [82, 0, 255],
      [163, 255, 0],
      [255, 235, 0],
      [8, 184, 170],
      [133, 0, 255],
      [0, 255, 92],
      [184, 0, 255],
      [255, 0, 31],
      [0, 184, 255],
      [0, 214, 255],
      [255, 0, 112],
      [92, 255, 0],
      [0, 224, 255],
      [112, 224, 255],
      [70, 184, 160],
      [163, 0, 255],
      [153, 0, 255],
      [71, 255, 0],
      [255, 0, 163],
      [255, 204, 0],
      [255, 0, 143],
      [0, 255, 235],
      [133, 255, 0],
      [255, 0, 235],
      [245, 0, 255],
      [255, 0, 122],
      [255, 245, 0],
      [10, 190, 212],
      [214, 255, 0],
      [0, 204, 255],
      [20, 0, 255],
      [255, 255, 0],
      [0, 153, 255],
      [0, 41, 255],
      [0, 255, 204],
      [41, 0, 255],
      [41, 255, 0],
      [173, 0, 255],
      [0, 245, 255],
      [71, 0, 255],
      [122, 0, 255],
      [0, 255, 184],
      [0, 92, 255],
      [184, 255, 0],
      [0, 133, 255],
      [255, 214, 0],
      [25, 194, 194],
      [102, 255, 0],
      [92, 0, 255],
  ])

#Assign colors to objects#  
def labelAde20k_to_color_image(label):
  """Adds color defined by the dataset colormap to the label.

  Args:
    label: A 2D array with integer type, storing the segmentation label.

  Returns:
    result: A 2D array with floating type. The element of the array
      is the color indexed by the corresponding element in the input label
      to the PASCAL color map.

  Raises:
    ValueError: If label is not of rank 2 or its value is larger than color
      map maximum entry.
  """
  if label.ndim != 2:
    raise ValueError('Expect 2-D input label')

  colormap = create_ade20k_label_colormap()

  if np.max(label) >= len(colormap):
    raise ValueError('label value too large.')

  return colormap[label]   

