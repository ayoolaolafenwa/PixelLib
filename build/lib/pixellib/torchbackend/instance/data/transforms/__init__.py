# Copyright (c) Facebook, Inc. and its affiliates.
import sys
#sys.path.insert(0, 'C:/Users/olafe/OneDrive/ModDectron/separate/PointRend/detectron2/utils')
from fvcore.transforms.transform import Transform, TransformList  # order them first
from fvcore.transforms.transform import *
from .transform import *
from .augmentation import *
from .augmentation_impl import *

__all__ = [k for k in globals().keys() if not k.startswith("_")]


from pixellib.torchbackend.instance.utils.env import fixup_module_metadata

fixup_module_metadata(__name__, globals(), __all__)
del fixup_module_metadata
