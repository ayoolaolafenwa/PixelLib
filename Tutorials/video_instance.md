# Instance segmentation of videos with PixelLib**

Instance segmentation with PixelLib is based on MaskRCNN framework.

Download the mask rcnn model from [here](https://github.com/ayoolaolafenwa/PixelLib/releases/download/1.2/mask_rcnn_coco.h5)

**Code to implement instance segmentation of videos**


```python

  import pixellib
  from pixellib.instance import instance_segmentation

  segment_video = instance_segmentation()
  segment_video.load_model("mask_rcnn_coco.h5")
  segment_video.process_video("video_path", frames_per_second= 20, output_video_name="path_to_outputvideo")

```
```python

  import pixellib
  from pixellib.instance 

  segment_video = instance_segmentation()
```

We imported in the class for performing instance segmentation and created an instance of the class.

```python
  
  segment_video.load_model("mask_rcnn_coco.h5")
```

We loaded the maskrcnn model trained on coco dataset to perform instance segmentation and it can be downloaded from here.

```python

  segment_video.process_video("sample_video2.mp4", frames_per_second = 20, output_video_name = "output_video.mp4")
```
We called the function  to perform segmentation on the video file.

It takes the following parameters:-

**video_path:** the path to the video file we want to perform segmentation on.

**frames_per_second:** this is parameter to set the number.of frames per second for the output video file. In this case it is set to 15 i.e the saved video file will have 15 frames per second.

**output_video_name:** the saved segmented video. The output video will be saved in your current working directory.  

**Sample video2**

<iframe width="560" height="315"
src="https://www.youtube.com/embed/EivIBccZURA" 
frameborder="0" 
allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" 
allowfullscreen></iframe>



```python

  import pixellib
  from pixellib.instance import instance_segmentation

  segment_video = instance_segmentation()
  segment_video.load_model("mask_rcnn_coco.h5")
  segment_video.process_video("sample_video2.mp4", frames_per_second= 15, output_video_name="output_video.mp4")
```
**Output video**

<iframe width="560" height="315"
src="https://www.youtube.com/embed/yu03363mlNM" 
frameborder="0" 
allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" 
allowfullscreen></iframe>



We can perform instance segmentation with object detection by setting the parameter *show_bboxes* to true.


```python

  import pixellib
  from pixellib.instance import instance_segmentation

  segment_video = instance_segmentation()
  segment_video.load_model("mask_rcnn_coco.h5")
  segment_video.process_video("sample_video2.mp4", show_bboxes = True, frames_per_second= 15, output_video_name="output_video.mp4")
```


**Output video with bounding boxes**

<iframe width="560" height="315"
src="https://www.youtube.com/embed/bGPO1bCZLAo" 
frameborder="0" 
allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" 
allowfullscreen></iframe>




# Instance Segmentation of Live Camera.

We can use the same model to perform semantic segmentation on camera. This can be done by few modifications to the code used to process video file.

```python

  import pixellib
  from pixellib.instance import instance_segmentation
  import cv2


  capture = cv2.VideoCapture(0)

  segment_video = instance_segmentation()
  segment_video.load_model("mask_rcnn_coco.h5")
  segment_video.process_camera(capture, frames_per_second= 10, output_video_name="output_video.mp4", show_frames= True,
  frame_name= "frame", check_fps = True)
```

```python

  import cv2 
  capture = cv2.VideoCapture(0)
```

We imported cv2 and included the code to capture camera frames.

```python

  segment_video.process_camera(capture, show_bboxes = True, frames_per_second = 15, output_video_name = "output_video.mp4, show_frames = True, frame_name = "frame")  
```

In the code for performing segmentation, we replaced the video filepath to capture i.e we are going to process a stream camera frames instead of a video file.We added extra parameters for the purpose of showing the camera frames.
  
**show_frames** this parameter handles showing of segmented camera frames and press q to exist.

**frame_name** this is the name given to the shown camera's frame.

**check_fps** You may want to check the number of frames processed, just set the parameter check_fps is true.It will print out the number of frames per seconds. In this case it is *30 frames per second*.


A demo showing the output of pixelLib's instance segmentation on camera's feeds using MASK-RCNN model.

<iframe width="560" height="315"
src="https://www.youtube.com/embed/HD1m-g7cOKw&list=PLtFkVrcr8LqNgbwdOb6of5X19ytm4ycHC&index=6&t=0s" 
frameborder="0" 
allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" 
allowfullscreen></iframe>