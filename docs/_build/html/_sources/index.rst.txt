PIXELLIB'S OFFICIAL DOCUMENTATION
====================================

PixelLib is a library created for performing image segmentation using few lines of code. It is a flexible library created to allow easy integration of image segmentation into software solutions.


PixelLib requires python's version 3.5-3.7, `Download python <https://www.python.org/downloads/>`_

It requires pip's version >= 19.0 

Install pip with:

.. code-block:: python

  pip3 install pip

Install PixelLib and its dependencies:

Install  the latest version of tensorflow(Tensorflow 2.0+) with:

.. code-block:: python

  pip3 install tensorflow


Install Opencv-python with:

.. code-block:: python

  pip3 install opencv-python

Install Pillow with:

.. code-block:: python

  pip3 install pillow

Install scikit-image with:

.. code-block:: python

  pip3 install scikit-image

Install PixelLib with:

.. code-block:: python

  pip3 install pixellib



PixelLib supports the two major types of segmentation: 

1 **Semantic segmentation**:
Objects in an image with the same pixel values are segmented with the same colormaps.

.. image:: photos/semantic.jpg

:ref:`semantic`

2 **Instance segmentation**:
Instances of the same object are segmented with different color maps.

.. image:: photos/instance.jpg

:ref:`instance`


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   semantic.rst
   instance.rst
   



.. Indices and tables
.. ==================

.. * :ref:`genindex`
.. * :ref:`modindex`
.. * :ref:`search`

  
