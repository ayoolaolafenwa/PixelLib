"""from pixellib.instance import instance_segmentation

ins = instance_segmentation(detection_speed="average")
#ins = instance_segmentation()
ins.load_model("mask_rcnn_coco.h5")
ins.segmentImage("sample4.jpg", show_bboxes=True, output_image_name="a1_fastest2.jpg")

"""#import pixellib

import pixellib
from pixellib.custom_train import instance_custom_training

a = instance_custom_training()
a.modelConfig(num_classes=2)
a.load_dataset("Nature")
a.load_pretrained_model("mask_rcnn_coco.h5")
a.train_model(num_epochs = 1, path_trained_models = "sample", layers="heads", augmentation=True) 
