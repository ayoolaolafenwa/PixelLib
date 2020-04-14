import tensorflow as tf
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib import gridspec
from .deeplab import Deeplab_xcep_pascal
from .deeplab import Deeplab_xcep_cityscapes
import numpy as np


class semantic_segmentation():
    def __init__(self):

      self.model3 = Deeplab_xcep_pascal()
      self.model4 = Deeplab_xcep_cityscapes()
      
      
    def load_pascalvoc_model(self, model_path):
      self.model3.load_weights(model_path)
    
    def load_cityscapes_model(self, model_path):
      self.model4.load_weights(model_path)

    def segmentAsPascalvoc(self, image_path, output_image_name, segmap_only = True):

      if segmap_only == True:
        trained_image_width=512
        mean_subtraction_value=127.5
        image = np.array(Image.open(image_path))

        # resize to max dimension of images from training dataset
        w, h, _ = image.shape
        ratio = float(trained_image_width) / np.max([w, h])
        resized_image = np.array(Image.fromarray(image.astype('uint8')).resize((int(ratio * h), int(ratio * w))))

        # apply normalization 
        resized_image = (resized_image / mean_subtraction_value) - 1.

        # pad array to square image to match training images
        pad_x = int(trained_image_width - resized_image.shape[0])
        pad_y = int(trained_image_width - resized_image.shape[1])
        resized_image = np.pad(resized_image, ((0, pad_x), (0, pad_y), (0, 0)), mode='constant')

        res = self.model3.predict(np.expand_dims(resized_image, 0))
        labels = np.argmax(res.squeeze(), -1)
        # remove padding and resize back to original image
        if pad_x > 0:
          labels = labels[:-pad_x]
        if pad_y > 0:
          labels = labels[:, :-pad_y]
        
        #Apply segmentation color map
        labels = labelP_to_color_image(labels)
        labels = np.array(Image.fromarray(labels.astype('uint8')).resize((h, w)))
        
        if output_image_name:

          plt.imsave(output_image_name, labels)
          plt.axis("off")
          output = plt.imshow(labels)   
          return output
    
      elif segmap_only == False:
        trained_image_width=512
        mean_subtraction_value=127.5
        image = np.array(Image.open(image_path))

        # resize to max dimension of images from training dataset
        w, h, _ = image.shape
        ratio = float(trained_image_width) / np.max([w, h])
        resized_image = np.array(Image.fromarray(image.astype('uint8')).resize((int(ratio * h), int(ratio * w))))

        # apply normalization 
        resized_image = (resized_image / mean_subtraction_value) - 1.

        # pad array to square image to match training images
        pad_x = int(trained_image_width - resized_image.shape[0])
        pad_y = int(trained_image_width - resized_image.shape[1])
        resized_image = np.pad(resized_image, ((0, pad_x), (0, pad_y), (0, 0)), mode='constant')

            
        res = self.model3.predict(np.expand_dims(resized_image, 0))
        labels = np.argmax(res.squeeze(), -1)
        # remove padding and resize back to original image
        if pad_x > 0:
          labels = labels[:-pad_x]
        if pad_y > 0:
          labels = labels[:, :-pad_y]
        
        
        labels = np.array(Image.fromarray(labels.astype('uint8')).resize((h, w)))
        #Apply segmentation color map
        labels = labelP_to_color_image(labels)
        if output_image_name:
          plt.figure(figsize=(15, 5))
          grid_spec = gridspec.GridSpec(1, 3, width_ratios=[6, 6, 6])

          plt.subplot(grid_spec[0])
          plt.imshow(image)
          plt.axis('off')
          plt.title('input image')

          plt.subplot(grid_spec[1])
          plt.imshow(image)
          plt.imshow(labels, alpha=0.7)
          plt.axis('off')
          plt.title('segmentation overlay')

          plt.subplot(grid_spec[2]) 
          plt.imshow(labels)
          plt.axis('off')
          plt.title('segmentation map')
      
          plt.grid('off')

          plt.savefig(output_image_name, bbox_inches = "tight")
          output = plt.show()

    
          return output

    def segmentAsCityscapes(self, image_path, output_image_name, segmap_only = True):

      if segmap_only == True:
        trained_image_width=512
        mean_subtraction_value=127.5
        image = np.array(Image.open(image_path))

        # resize to max dimension of images from training dataset
        w, h, _ = image.shape
        ratio = float(trained_image_width) / np.max([w, h])
        resized_image = np.array(Image.fromarray(image.astype('uint8')).resize((int(ratio * h), int(ratio * w))))

        # apply normalization 
        resized_image = (resized_image / mean_subtraction_value) - 1.

        # pad array to square image to match training images
        pad_x = int(trained_image_width - resized_image.shape[0])
        pad_y = int(trained_image_width - resized_image.shape[1])
        resized_image = np.pad(resized_image, ((0, pad_x), (0, pad_y), (0, 0)), mode='constant')

        res = self.model4.predict(np.expand_dims(resized_image, 0))
        labels = np.argmax(res.squeeze(), -1)
        # remove padding and resize back to original image
        if pad_x > 0:
          labels = labels[:-pad_x]
        if pad_y > 0:
          labels = labels[:, :-pad_y]
        
        #Apply segmentation color map
        labels = labelC_to_color_image(labels)
        labels = np.array(Image.fromarray(labels.astype('uint8')).resize((h, w)))
        
        if output_image_name:
          plt.subplots(1, figsize = (15,15))
          plt.axis("off")
          plt.imshow(image)
          plt.imshow(labels, alpha = 0.7)
          plt.savefig(output_image_name, bbox_inches = "tight", pad_inches = 0)
          output = plt.show()
          return output

          
    
      elif segmap_only == False:
        trained_image_width=512
        mean_subtraction_value=127.5
        image = np.array(Image.open(image_path))

        # resize to max dimension of images from training dataset
        w, h, _ = image.shape
        ratio = float(trained_image_width) / np.max([w, h])
        resized_image = np.array(Image.fromarray(image.astype('uint8')).resize((int(ratio * h), int(ratio * w))))

        # apply normalization for trained dataset images
        resized_image = (resized_image / mean_subtraction_value) - 1.

        # pad array to square image to match training images
        pad_x = int(trained_image_width - resized_image.shape[0])
        pad_y = int(trained_image_width - resized_image.shape[1])
        resized_image = np.pad(resized_image, ((0, pad_x), (0, pad_y), (0, 0)), mode='constant')

            
        res = self.model4.predict(np.expand_dims(resized_image, 0))
        labels = np.argmax(res.squeeze(), -1)
        # remove padding and resize back to original image
        if pad_x > 0:
          labels = labels[:-pad_x]
        if pad_y > 0:
          labels = labels[:, :-pad_y]
        
        
        labels = np.array(Image.fromarray(labels.astype('uint8')).resize((h, w)))
        #Apply segmentation color map
        labels = labelC_to_color_image(labels)
        if output_image_name:
          
          plt.figure(figsize=(15, 15))
          grid_spec = gridspec.GridSpec(1, 2, width_ratios=[6, 6])

          plt.subplot(grid_spec[0])
          plt.imshow(image)
          plt.axis('off')
          plt.title('input image')

          plt.subplot(grid_spec[1])
          plt.imshow(image)
          plt.imshow(labels, alpha=0.7)
          plt.axis('off')
          plt.title('output image')

          plt.savefig(output_image_name, bbox_inches = "tight", pad_inches = 0)
          output = plt.show()
        

          return output

    
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




def create_cityscapes_label_colormap():
  """Creates a label colormap used in CITYSCAPES segmentation benchmark.
  Returns:
    A colormap for visualizing segmentation results.
  """
  colormap = np.zeros((256, 3), dtype=np.uint8)
  colormap[0] = [128, 64, 128]
  colormap[1] = [244, 35, 232]
  colormap[2] = [70, 70, 70]
  colormap[3] = [102, 102, 156]
  colormap[4] = [190, 153, 153]
  colormap[5] = [153, 153, 153]
  colormap[6] = [250, 170, 30]
  colormap[7] = [220, 220, 0]
  colormap[8] = [107, 142, 35]
  colormap[9] = [152, 251, 152]
  colormap[10] = [70, 130, 180]
  colormap[11] = [220, 20, 60]
  colormap[12] = [255, 0, 0]
  colormap[13] = [0, 0, 142]
  colormap[14] = [0, 0, 70]
  colormap[15] = [0, 60, 100]
  colormap[16] = [0, 80, 100]
  colormap[17] = [0, 0, 230]
  colormap[18] = [119, 11, 32]
  return colormap


def labelC_to_color_image(label):
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

  colormap = create_cityscapes_label_colormap()

  if np.max(label) >= len(colormap):
    raise ValueError('label value too large.')

  return colormap[label]
