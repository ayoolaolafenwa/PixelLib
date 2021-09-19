# Copyright (c) Facebook, Inc. and its affiliates.
from pixellib.torchbackend.instance.layers.shape_spec import ShapeSpec

from pixellib.torchbackend.instance.modeling.anchor_generator import build_anchor_generator, ANCHOR_GENERATOR_REGISTRY
from pixellib.torchbackend.instance.modeling.backbone.build import BACKBONE_REGISTRY
from pixellib.torchbackend.instance.modeling.backbone.fpn import FPN
from pixellib.torchbackend.instance.modeling.backbone import backbone
from pixellib.torchbackend.instance.modeling.backbone.resnet import ResNet, ResNetBlockBase
from pixellib.torchbackend.instance.modeling.backbone.build import build_backbone 
from pixellib.torchbackend.instance.modeling.backbone.resnet import build_resnet_backbone
from pixellib.torchbackend.instance.modeling.backbone.backbone import Backbone
from pixellib.torchbackend.instance.modeling.backbone.fpn import FPN
from pixellib.torchbackend.instance.modeling.backbone.resnet import ResNet, ResNetBlockBase, build_resnet_backbone
from pixellib.torchbackend.instance.modeling.backbone.resnet import make_stage

'''from pixellib.torchbackend.instance.modeling.backbone.backbone import (
    BACKBONE_REGISTRY,
    FPN,
    Backbone,
    ResNet,
    ResNetBlockBase,
    build_backbone,
    build_resnet_backbone,
    make_stage,
)
'''

from pixellib.torchbackend.instance.modeling.meta_arch.build import META_ARCH_REGISTRY
from pixellib.torchbackend.instance.modeling.meta_arch import (
    META_ARCH_REGISTRY,
    SEM_SEG_HEADS_REGISTRY,
    GeneralizedRCNN,
    #PanopticFPN,
    ProposalNetwork,
    #RetinaNet,
    #SemanticSegmentor,
    build_model,
    build_sem_seg_head,
)

from .postprocessing import detector_postprocess
from pixellib.torchbackend.instance.modeling.proposal_generator import (
    PROPOSAL_GENERATOR_REGISTRY,
    build_proposal_generator,
    RPN_HEAD_REGISTRY,
    build_rpn_head,
)
from pixellib.torchbackend.instance.modeling.roi_heads import (
    ROI_BOX_HEAD_REGISTRY,
    ROI_HEADS_REGISTRY,
    ROI_KEYPOINT_HEAD_REGISTRY,
    ROI_MASK_HEAD_REGISTRY,
    ROIHeads,
    StandardROIHeads,
    BaseMaskRCNNHead,
    BaseKeypointRCNNHead,
    FastRCNNOutputLayers,
    build_box_head,
    build_keypoint_head,
    build_mask_head,
    build_roi_heads,
)
from .test_time_augmentation import DatasetMapperTTA, GeneralizedRCNNWithTTA
from .mmdet_wrapper import MMDetBackbone, MMDetDetector

_EXCLUDE = {"ShapeSpec"}
__all__ = [k for k in globals().keys() if k not in _EXCLUDE and not k.startswith("_")]


from pixellib.torchbackend.instance.utils.env import fixup_module_metadata

fixup_module_metadata(__name__, globals(), __all__)
del fixup_module_metadata
