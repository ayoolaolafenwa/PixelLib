import tensorflow as tf
import numpy as np
from PIL import Image
from .deeplab import Deeplab_xcep_pascal
import numpy as np
import cv2


class semantic_segmentation():

  def __init__(self):

    self.model = Deeplab_xcep_pascal()
    
      
  def load_pascalvoc_model(self, model_path):
    self.model.load_weights(model_path)


  def segmentAsPascalvoc(self, image_path, output_image_name=None,overlay=False):            
    trained_image_width=512
    mean_subtraction_value=127.5
    image = np.array(Image.open(image_path))     
    output = image.copy()

    # resize to max dimension of images from training dataset
    w, h, _ = image.shape
    ratio = float(trained_image_width) / np.max([w, h])
    resized_image = np.array(Image.fromarray(image.astype('uint8')).resize((int(ratio * h), int(ratio * w))))
    resized_image = (resized_image / mean_subtraction_value) -1


    # pad array to square image to match training images
    pad_x = int(trained_image_width - resized_image.shape[0])
    pad_y = int(trained_image_width - resized_image.shape[1])
    resized_image = np.pad(resized_image, ((0, pad_x), (0, pad_y), (0, 0)), mode='constant')

    print("Processing image....")

    #run prediction
    res = self.model.predict(np.expand_dims(resized_image, 0))
    
    labels = np.argmax(res.squeeze(), -1)
    # remove padding and resize back to original image
    if pad_x > 0:
        labels = labels[:-pad_x]
    if pad_y > 0:
        labels = labels[:, :-pad_y]
        
    #Apply segmentation color map
    labels = labelP_to_color_image(labels)   
    labels = np.array(Image.fromarray(labels.astype('uint8')).resize((h, w)))
    
    
    new_img = cv2.cvtColor(labels, cv2.COLOR_RGB2BGR)

    if overlay == True:
        alpha = 0.7
        cv2.addWeighted(new_img, alpha, output, 1 - alpha,0, output)

        if output_image_name is not None:
          cv2.imwrite(output_image_name, output)
          print("Processed Image saved successfully in your current working directory.")

        return new_img, output

        
    else:  
        if output_image_name is not None:
  
          cv2.imwrite(output_image_name, new_img)

          print("Processed Image saved successfuly in your current working directory.")

        return new_img, None
    
    

    
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




