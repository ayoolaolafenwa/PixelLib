# PixelLib
Pixellib is a library for performing segmentation of images. It supports the two major types of image segmentation: 
## 1.Semantic segmentation
## 2.Instance segmentation
You can implement both semantic and instance segmentation with few lines of code.

# Install PixelLib and its dependencies:

Install latest version of tensorflow(Tensorflow 2.0+) with:

**pip3 install tensorflow**

Install Matplotlib with:
**pip3 install matplotlib**

Install Pillow with:
**pip3 install pillow**

Install Scipy with:
**pip3 install scipy**

Install Ipython with:
**pip3 install Ipython**

Install Skimage with:
**pip3 install scikit-image**

Install h5py with:
**pip3 install h5py**

## Install Pixellib with:
**pip3 install pixellib**

# SEMANTIC SEGMENTATION WITH PIXELLIB:
Pixellib is implemented with Deeplabv3+ framework to perform semantic segmentation.  Xception models pretrained on pascalvoc and cityscapes datasets are used for semantic segmentation. 

## Semantic segmentation with xception model pretrained on pascalvoc.
```python
from pixellib.semantic import semantic_segmentation

segment_image = semantic_segmentation()
segment_image.load_pascalvoc_model("deeplabv3_xception_tf_dim_ordering_tf_kernels.h5") 
segment_image.segmentAsPascalvoc("sep_mod.jpg", output_image_name = "p3.jpg", segmap_only = False)

```
We shall take a look into 
