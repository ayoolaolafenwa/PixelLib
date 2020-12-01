.. _video_pascal:

**Semantic Segmentation of videos with PixelLib using Pascalvoc model**
========================================================================

PixelLib is implemented with Deeplabv3+ framework to perform semantic segmentation. Xception model trained on pascalvoc dataset is used for semantic segmentation.

Download the xception model from `here <https://github.com/ayoolaolafenwa/PixelLib/releases/download/1.1/deeplabv3_xception_tf_dim_ordering_tf_kernels.h5>`_.

**Code to implement semantic segmentation of a video with pascalvoc model**:

.. code-block:: python



  import pixellib
  from pixellib.semantic import semantic_segmentation

  segment_video = semantic_segmentation()
  segment_video.load_pascalvoc_model("deeplabv3_xception_tf_dim_ordering_tf_kernels.h5")
  segment_video.process_video_pascalvoc("video_path",  overlay = True, frames_per_second= 15, output_video_name="path_to_output_video")

We shall take a look into each line of code.


.. code-block:: python

  import pixellib
  from pixellib.semantic import semantic_segmentation

  #created an instance of semantic segmentation class
  segment_image = semantic_segmentation()

The class for performing semantic segmentation is imported from pixellib and we created an instance of the class.

.. code-block:: python

  segment_image.load_pascalvoc_model("deeplabv3_xception_tf_dim_ordering_tf_kernels.h5") 

We called the function to load the xception model trained on pascal voc. 

.. code-block:: python

  segment_video.process_video_pascalvoc("video_path",  overlay = True, frames_per_second= 15, output_video_name="path_to_output_video")

This is the line of code that performs segmentation on an image and the segmentation is done in the pascalvoc's color format. This function takes in two parameters:

*video_path:* the path to the video file we want to perform segmentation on.

*frames_per_second:* this is parameter to set the number of frames per second for the output video file. In this case it is set to 15 i.e the saved video file will have 15 frames per second.

*output_video_name:* the saved segmented video. The output video will be saved in your current working directory.

**sample_video**

.. raw:: html

    <div style="position: relative; padding-bottom: 56.25%; height: 0; overflow: hidden; max-width: 100%; height: auto;">
        <iframe src="https://www.youtube.com/embed/8fkthbwqmB0" frameborder="0" allowfullscreen style="position: absolute; top: 0; left: 0; width: 100%; height: 100%;"></iframe>
    </div>


.. code-block:: python

  import pixellib
  from pixellib.semantic import semantic_segmentation

  segment_video = semantic_segmentation()
  segment_video.load_pascalvoc_model("deeplabv3_xception_tf_dim_ordering_tf_kernels.h5")
  segment_video.process_video_pascalvoc("sample_video1.mp4",  overlay = True, frames_per_second= 15, output_video_name="output_video.mp4")

**Output video**

.. raw:: html

    <div style="position: relative; padding-bottom: 56.25%; height: 0; overflow: hidden; max-width: 100%; height: auto;">
        <iframe src="https://www.youtube.com/embed/l9WMqT2znJE" frameborder="0" allowfullscreen style="position: absolute; top: 0; left: 0; width: 100%; height: 100%;"></iframe>
    https://www.youtube.com/watch?v=_NsELe67UxM
    </div>

This is a saved segmented video using pascal voc model.

**Segmentation of live camera with pascalvoc model**
====================================================


We can use the same model to perform semantic segmentation on camera. This can be done by few modifications to the code to process video file.

.. code-block:: python

  import pixellib
  from pixellib.semantic import semantic_segmentation
  import cv2


  capture = cv2.VideoCapture(0)

  segment_video = semantic_segmentation()
  segment_video.load_pascalvoc_model("deeplabv3_xception_tf_dim_ordering_tf_kernels.h5")
  segment_video.process_camera_pascalvoc(capture,  overlay = True, frames_per_second= 15, output_video_name="output_video.mp4", show_frames= True,
  frame_name= "frame")


We imported cv2 and included the code to capture camera's frames.

.. code-block:: python

  segment_video.process_camera_pascalvoc(capture,  overlay = True, frames_per_second= 15, output_video_name="output_video.mp4", show_frames= True,frame_name= "frame")  


In the code for performing segmentation, we replaced the video's filepath to capture i.e we are going to process a stream camera's frames instead of a video file.We added extra parameters for the purpose of showing the camera frames:

*show_frames:* this parameter handles showing of segmented camera frames and press q to exist.
*frame_name:* this is the name given to the shown camera's frames.




.. raw:: html

    <div style="position: relative; padding-bottom: 56.25%; height: 0; overflow: hidden; max-width: 100%; height: auto;">
        <iframe src="https://www.youtube.com/embed/8oSRYf9Ow2E" frameborder="0" allowfullscreen style="position: absolute; top: 0; left: 0; width: 100%; height: 100%;"></iframe>
    </div>


A demo showing the output of pixelLib’s semantic segmentation of camera’s feeds using pascal voc model.
*Good work! It was able to successfully segment me and the plastic bottle in front of me.*

