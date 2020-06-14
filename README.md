![alt_test1](instance_mask/cover.jpg)
# PixelLib 

[![Downloads](https://pepy.tech/badge/pixellib)](https://pepy.tech/project/pixellib)  [![Downloads](https://pepy.tech/badge/pixellib/month)](https://pepy.tech/project/pixellib/month)  [![Downloads](https://pepy.tech/badge/pixellib/week)](https://pepy.tech/project/pixellib/week)

Pixellib is a library for performing segmentation of images. It supports the two major types of image segmentation: 

**1.Semantic segmentation**

**2.Instance segmentation**

You can implement both semantic and instance segmentation with few lines of code.

There are two types of Deeplabv3+ models available for performing **semantic segmentation** with PixelLib:

1. Deeplabv3+ model with xception as network backbone trained on Ade20k dataset, a dataset with 150 classes of objects.
2. Deeplabv3+ model with xception as network backbone trained on Pascalvoc dataset, a dataset with 20 classes of objects. 

Instance segmentation is implemented with PixelLib by using Mask R-CNN model trained on coco dataset.

**Note** Deeplab and mask r-ccn models are available  in the [release](https://github.com/ayoolaolafenwa/PixelLib/releases) of this repository.

# Install Tensorflow:

Install latest version of tensorflow(Tensorflow 2.0+) with:

**pip3 install tensorflow**


## Install Pixellib with:
**pip3 install pixellib --upgrade**

* Check out tutorials on PixelLib on [medium](https://medium.com/@olafenwaayoola/image-segmentation-with-six-lines-0f-code-acb870a462e8)and documentation on [readthedocs](https://pixellib.readthedocs.io/en/latest/)


## Read the following tutorials on performing both semantic and instance segmentation with PixelLib.

![alt_test1](Images/ade_cover.jpg)
# [Tutorial On Semantic Segmentation of Images With PixelLib Using Ade20k model](Tutorials/image_ade20k.md)



![alt_test3](Images/pascal.jpg)
# [Tutorial OnSemantic Segmentation of Images With PixelLib Using Pascalvoc model](Tutorials/image_pascalvoc.md)

![alt_test1](instance_mask/result2.jpg)
# [Tutorial On Instance Segmentation of Images With PixelLib Using Mask-RCNN](Tutorials/image_instance.md)



## References
1. Bonlime, Keras implementation of Deeplab v3+ with pretrained weights  https://github.com/bonlime/keras-deeplab-v3-plus

2. Liang-Chieh Chen. et al, Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation https://arxiv.org/abs/1802.02611

3. Matterport, Mask R-CNN for object detection and instance segmentation on Keras and TensorFlow https://github.com/matterport/Mask_RCNN

4. Kaiming He et al, Mask R-CNN https://arxiv.org/abs/1703.06870

[Back To Top](#pixellib)
