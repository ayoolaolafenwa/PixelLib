#SEMANTIC SEGMENTATION

import pixellib
from pixellib.semantic import semantic_segmentation

segment = semantic_segmentation()
segment.load_pascalvoc_model("pascal.h5")
out = segment.segmentAsPascalvoc("t1.jpg", output_image_name = "image1.jpg")

"""#Apply segmentation overlay
import pixellib
from pixellib.semantic import semantic_segmentation

segment = semantic_segmentation()
segment.load_pascalvoc_model("pascal.h5")
out = segment.segmentAsPascalvoc("t1.jpg", output_image_name = "image2.jpg", overlay=True)


#INSTANCE SEGMENTATION
import pixellib
from pixellib.instance import instance_segmentation

instance_seg = instance_segmentation()
instance_seg.load_model("mask_rcnn_coco.h5")
out = instance_seg.segmentImage("img.jpg", output_image_name = "image3.jpg")

#Show segmented images with bounding boxes
import pixellib
from pixellib.instance import instance_segmentation

instance_seg = instance_segmentation()
instance_seg.load_model("mask_rcnn_coco.h5")
out = instance_seg.segmentImage("img.jpg", output_image_name = "image4.jpg", show_bboxes = True)"""