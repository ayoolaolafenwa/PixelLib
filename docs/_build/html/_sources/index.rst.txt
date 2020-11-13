PIXELLIB'S OFFICIAL DOCUMENTATION
====================================

PixelLib is a library created for performing image and video segmentation using few lines of code. It is a flexible library created to allow easy integration of image and video segmentation into software solutions.


PixelLib requires python's version 3.5-3.7, `Download python <https://www.python.org/downloads/>`_

It requires pip's version >= 19.0 

Install pip with:

.. code-block:: python

  pip3 install pip

Install PixelLib and its dependencies:

Install  the latest version of tensorflow(Tensorflow 2.0+) with:

.. code-block:: python

  pip3 install tensorflow

Install imgaug with:

.. code-block:: python

  pip3 install imgaug

Install PixelLib with:

.. code-block:: python

  pip3 install pixellib --upgrade



PixelLib supports the two major types of segmentation and you can create a custom model for objects' segmentation by training your dataset with PixelLib: 


1 **Semantic segmentation**:
Objects in an image with the same pixel values are segmented with the same colormaps.

.. image:: photos/ade_overlay.jpg

:ref:`semantic_ade20k`

.. raw:: html

    <div style="position: relative; padding-bottom: 56.25%; height: 0; overflow: hidden; max-width: 100%; height: auto;">
        <iframe src="https://www.youtube.com/embed/hxczTe9U8jY" frameborder="0" allowfullscreen style="position: absolute; top: 0; left: 0; width: 100%; height: 100%;"></iframe>
    https://www.youtube.com/watch?v=_NsELe67UxM
    </div>


:ref:`video_ade20k`

.. image:: photos/pascal.jpg

:ref:`image_pascal`

.. raw:: html

    <div style="position: relative; padding-bottom: 56.25%; height: 0; overflow: hidden; max-width: 100%; height: auto;">
        <iframe src="https://www.youtube.com/embed/l9WMqT2znJE" frameborder="0" allowfullscreen style="position: absolute; top: 0; left: 0; width: 100%; height: 100%;"></iframe>
    https://www.youtube.com/watch?v=_NsELe67UxM
    </div>

:ref:`video_pascal`




2 **Instance segmentation**:
Instances of the same object are segmented with different color maps.

.. image:: photos/ins.jpg


:ref:`image_instance`


.. raw:: html

    <div style="position: relative; padding-bottom: 56.25%; height: 0; overflow: hidden; max-width: 100%; height: auto;">
        <iframe src="https://www.youtube.com/embed/bGPO1bCZLAo" frameborder="0" allowfullscreen style="position: absolute; top: 0; left: 0; width: 100%; height: 100%;"></iframe>
    
    </div>

:ref:`video_instance`


3 **Implement Instance Segmentation And Object Detection On Objects By Training Your Dataset..**

.. image:: photos/cover.jpg

:ref:`custom_train`


**Inference With A Custom Model Trained With PixelLib**

.. raw:: html

    <div style="position: relative; padding-bottom: 56.25%; height: 0; overflow: hidden; max-width: 100%; height: auto;">
        <iframe src="https://www.youtube.com/embed/bWQGxaZIPOo" ,  frameborder="0" allowfullscreen style="position: absolute; top: 0; left: 0; width: 100%; height: 100%;"></iframe>
    https://www.youtube.com/watch?v=bWQGxaZIPOo
    </div>

:ref:`custom_inference`


**Implement background editing in images and videos using five lines of code. These are the features supported for background editing.**

**Change the background of an image with a picture**

**Assign a distinct color to the background of an image**

**Grayscale the background of an image**

**Blur the background of an image**


**Change the background of an Image**

.. image:: photos/bg_cover.jpg

:ref:`image_bg`


**Change the background of a Video**

.. raw:: html

    <div style="position: relative; padding-bottom: 56.25%; height: 0; overflow: hidden; max-width: 100%; height: auto;">
        <iframe src= "https://www.youtube.com/embed/duTiKf76ZU8", frameborder="0" allowfullscreen style="position: absolute; top: 0; left: 0; width: 100%; height: 100%;"></iframe>
    </div>

:ref:`video_bg`



.. toctree::
   :maxdepth: 4
   :caption: Contents:
   
   image_ade20k.rst
   video_ade20k.rst
   Image_pascal.rst
   video_pascal.rst
   Image_instance.rst
   video_instance.rst
   custom_train.rst
   custom_inference.rst
   change_image_bg.rst
   change_video_bg.rst
   
   

*CONTACT INFO:*

`olafenwaayoola@gmail.com <https://mail.google.com/mail/u/0/#inbox>`_

`Github.com <https://github.com/ayoolaolafenwa>`_

`Twitter.com <https://twitter.com/AyoolaOlafenwa>`_

`Facebook.com <https://web.facebook.com/ayofen?ref=bookmarks>`_

`Linkedin.com <https://www.linkedin.com/in/ayoola-olafenwa-003b901a9/>`_

.. Indices and tables
.. ==================

.. * :ref:`genindex`
.. * :ref:`modindex`
.. * :ref:`search`

  
