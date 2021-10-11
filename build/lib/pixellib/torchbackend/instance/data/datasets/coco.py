# Copyright (c) Facebook, Inc. and its affiliates.
import contextlib
import torch
import datetime
import io
import json
import logging
import numpy as np
import os
import shutil
import pycocotools.mask as mask_util
from fvcore.common.timer import Timer
from iopath.common.file_io import file_lock
from PIL import Image
from pixellib.torchbackend.instance.structures.masks import PolygonMasks
from pixellib.torchbackend.instance.structures.boxes import Boxes, BoxMode
from typing import List, Tuple
from pixellib.torchbackend.instance.utils.file_io import PathManager
from pixellib.torchbackend.instance.structures.boxes import _maybe_jit_unused
from pixellib.torchbackend.instance.data.catalogdata import MetadataCatalog, DatasetCatalog


"""
This file contains functions to parse COCO-format annotations into dicts in "Detectron2 format".
"""


logger = logging.getLogger(__name__)

__all__ = ["load_coco_json", "load_sem_seg", "convert_to_coco_json", "register_coco_instances"]


def load_coco_json(json_file, image_root, dataset_name=None, extra_annotation_keys=None):
    """
    Load a json file with COCO's instances annotation format.
    Currently supports instance detection, instance segmentation,
    and person keypoints annotations.

    Args:
        json_file (str): full path to the json file in COCO instances annotation format.
        image_root (str or path-like): the directory where the images in this json file exists.
        dataset_name (str or None): the name of the dataset (e.g., coco_2017_train).
            When provided, this function will also do the following:

            * Put "thing_classes" into the metadata associated with this dataset.
            * Map the category ids into a contiguous range (needed by standard dataset format),
              and add "thing_dataset_id_to_contiguous_id" to the metadata associated
              with this dataset.

            This option should usually be provided, unless users need to load
            the original json content and apply more processing manually.
        extra_annotation_keys (list[str]): list of per-annotation keys that should also be
            loaded into the dataset dict (besides "iscrowd", "bbox", "keypoints",
            "category_id", "segmentation"). The values for these keys will be returned as-is.
            For example, the densepose annotations are loaded in this way.

    Returns:
        list[dict]: a list of dicts in Detectron2 standard dataset dicts format (See
        `Using Custom Datasets </tutorials/datasets.html>`_ ) when `dataset_name` is not None.
        If `dataset_name` is None, the returned `category_ids` may be
        incontiguous and may not conform to the Detectron2 standard format.

    Notes:
        1. This function does not read the image files.
           The results do not have the "image" field.
    """
    from pycocotools.coco import COCO

    timer = Timer()
    json_file = PathManager.get_local_path(json_file)
    with contextlib.redirect_stdout(io.StringIO()):
        coco_api = COCO(json_file)
    if timer.seconds() > 1:
        logger.info("Loading {} takes {:.2f} seconds.".format(json_file, timer.seconds()))

    id_map = None
    if dataset_name is not None:
        meta = MetadataCatalog.get(dataset_name)
        cat_ids = sorted(coco_api.getCatIds())
        cats = coco_api.loadCats(cat_ids)
        # The categories in a custom json file may not be sorted.
        thing_classes = [c["name"] for c in sorted(cats, key=lambda x: x["id"])]
        meta.thing_classes = thing_classes

        # In COCO, certain category ids are artificially removed,
        # and by convention they are always ignored.
        # We deal with COCO's id issue and translate
        # the category ids to contiguous ids in [0, 80).

        # It works by looking at the "categories" field in the json, therefore
        # if users' own json also have incontiguous ids, we'll
        # apply this mapping as well but print a warning.
        if not (min(cat_ids) == 1 and max(cat_ids) == len(cat_ids)):
            if "coco" not in dataset_name:
                logger.warning(
                    """
Category ids in annotations are not in [1, #categories]! We'll apply a mapping for you.
"""
                )
        id_map = {v: i for i, v in enumerate(cat_ids)}
        meta.thing_dataset_id_to_contiguous_id = id_map

    # sort indices for reproducible results
    img_ids = sorted(coco_api.imgs.keys())
    # imgs is a list of dicts, each looks something like:
    # {'license': 4,
    #  'url': 'http://farm6.staticflickr.com/5454/9413846304_881d5e5c3b_z.jpg',
    #  'file_name': 'COCO_val2014_000000001268.jpg',
    #  'height': 427,
    #  'width': 640,
    #  'date_captured': '2013-11-17 05:57:24',
    #  'id': 1268}
    imgs = coco_api.loadImgs(img_ids)
    # anns is a list[list[dict]], where each dict is an annotation
    # record for an object. The inner list enumerates the objects in an image
    # and the outer list enumerates over images. Example of anns[0]:
    # [{'segmentation': [[192.81,
    #     247.09,
    #     ...
    #     219.03,
    #     249.06]],
    #   'area': 1035.749,
    #   'iscrowd': 0,
    #   'image_id': 1268,
    #   'bbox': [192.81, 224.8, 74.73, 33.43],
    #   'category_id': 16,
    #   'id': 42986},
    #  ...]
    anns = [coco_api.imgToAnns[img_id] for img_id in img_ids]
    total_num_valid_anns = sum([len(x) for x in anns])
    total_num_anns = len(coco_api.anns)
    if total_num_valid_anns < total_num_anns:
        logger.warning(
            f"{json_file} contains {total_num_anns} annotations, but only "
            f"{total_num_valid_anns} of them match to images in the file."
        )

    if "minival" not in json_file:
        # The popular valminusminival & minival annotations for COCO2014 contain this bug.
        # However the ratio of buggy annotations there is tiny and does not affect accuracy.
        # Therefore we explicitly white-list them.
        ann_ids = [ann["id"] for anns_per_image in anns for ann in anns_per_image]
        assert len(set(ann_ids)) == len(ann_ids), "Annotation ids in '{}' are not unique!".format(
            json_file
        )

    imgs_anns = list(zip(imgs, anns))
    logger.info("Loaded {} images in COCO format from {}".format(len(imgs_anns), json_file))

    dataset_dicts = []

    ann_keys = ["iscrowd", "bbox", "keypoints", "category_id"] + (extra_annotation_keys or [])

    num_instances_without_valid_segmentation = 0

    for (img_dict, anno_dict_list) in imgs_anns:
        record = {}
        record["file_name"] = os.path.join(image_root, img_dict["file_name"])
        record["height"] = img_dict["height"]
        record["width"] = img_dict["width"]
        image_id = record["image_id"] = img_dict["id"]

        objs = []
        for anno in anno_dict_list:
            # Check that the image_id in this annotation is the same as
            # the image_id we're looking at.
            # This fails only when the data parsing logic or the annotation file is buggy.

            # The original COCO valminusminival2014 & minival2014 annotation files
            # actually contains bugs that, together with certain ways of using COCO API,
            # can trigger this assertion.
            assert anno["image_id"] == image_id

            assert anno.get("ignore", 0) == 0, '"ignore" in COCO json file is not supported.'

            obj = {key: anno[key] for key in ann_keys if key in anno}
            if "bbox" in obj and len(obj["bbox"]) == 0:
                raise ValueError(
                    f"One annotation of image {image_id} contains empty 'bbox' value! "
                    "This json does not have valid COCO format."
                )

            segm = anno.get("segmentation", None)
            if segm:  # either list[list[float]] or dict(RLE)
                if isinstance(segm, dict):
                    if isinstance(segm["counts"], list):
                        # convert to compressed RLE
                        segm = mask_util.frPyObjects(segm, *segm["size"])
                else:
                    # filter out invalid polygons (< 3 points)
                    segm = [poly for poly in segm if len(poly) % 2 == 0 and len(poly) >= 6]
                    if len(segm) == 0:
                        num_instances_without_valid_segmentation += 1
                        continue  # ignore this instance
                obj["segmentation"] = segm

            keypts = anno.get("keypoints", None)
            if keypts:  # list[int]
                for idx, v in enumerate(keypts):
                    if idx % 3 != 2:
                        # COCO's segmentation coordinates are floating points in [0, H or W],
                        # but keypoint coordinates are integers in [0, H-1 or W-1]
                        # Therefore we assume the coordinates are "pixel indices" and
                        # add 0.5 to convert to floating point coordinates.
                        keypts[idx] = v + 0.5
                obj["keypoints"] = keypts

            obj["bbox_mode"] = BoxMode.XYWH_ABS
            if id_map:
                annotation_category_id = obj["category_id"]
                try:
                    obj["category_id"] = id_map[annotation_category_id]
                except KeyError as e:
                    raise KeyError(
                        f"Encountered category_id={annotation_category_id} "
                        "but this id does not exist in 'categories' of the json file."
                    ) from e
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)

    if num_instances_without_valid_segmentation > 0:
        logger.warning(
            "Filtered out {} instances without valid segmentation. ".format(
                num_instances_without_valid_segmentation
            )
            + "There might be issues in your dataset generation process.  Please "
            "check https://detectron2.readthedocs.io/en/latest/tutorials/datasets.html carefully"
        )
    return dataset_dicts


def load_sem_seg(gt_root, image_root, gt_ext="png", image_ext="jpg"):
    """
    Load semantic segmentation datasets. All files under "gt_root" with "gt_ext" extension are
    treated as ground truth annotations and all files under "image_root" with "image_ext" extension
    as input images. Ground truth and input images are matched using file paths relative to
    "gt_root" and "image_root" respectively without taking into account file extensions.
    This works for COCO as well as some other datasets.

    Args:
        gt_root (str): full path to ground truth semantic segmentation files. Semantic segmentation
            annotations are stored as images with integer values in pixels that represent
            corresponding semantic labels.
        image_root (str): the directory where the input images are.
        gt_ext (str): file extension for ground truth annotations.
        image_ext (str): file extension for input images.

    Returns:
        list[dict]:
            a list of dicts in detectron2 standard format without instance-level
            annotation.

    Notes:
        1. This function does not read the image and ground truth files.
           The results do not have the "image" and "sem_seg" fields.
    """

    # We match input images with ground truth based on their relative filepaths (without file
    # extensions) starting from 'image_root' and 'gt_root' respectively.
    def file2id(folder_path, file_path):
        # extract relative path starting from `folder_path`
        image_id = os.path.normpath(os.path.relpath(file_path, start=folder_path))
        # remove file extension
        image_id = os.path.splitext(image_id)[0]
        return image_id

    input_files = sorted(
        (os.path.join(image_root, f) for f in PathManager.ls(image_root) if f.endswith(image_ext)),
        key=lambda file_path: file2id(image_root, file_path),
    )
    gt_files = sorted(
        (os.path.join(gt_root, f) for f in PathManager.ls(gt_root) if f.endswith(gt_ext)),
        key=lambda file_path: file2id(gt_root, file_path),
    )

    assert len(gt_files) > 0, "No annotations found in {}.".format(gt_root)

    # Use the intersection, so that val2017_100 annotations can run smoothly with val2017 images
    if len(input_files) != len(gt_files):
        logger.warn(
            "Directory {} and {} has {} and {} files, respectively.".format(
                image_root, gt_root, len(input_files), len(gt_files)
            )
        )
        input_basenames = [os.path.basename(f)[: -len(image_ext)] for f in input_files]
        gt_basenames = [os.path.basename(f)[: -len(gt_ext)] for f in gt_files]
        intersect = list(set(input_basenames) & set(gt_basenames))
        # sort, otherwise each worker may obtain a list[dict] in different order
        intersect = sorted(intersect)
        logger.warn("Will use their intersection of {} files.".format(len(intersect)))
        input_files = [os.path.join(image_root, f + image_ext) for f in intersect]
        gt_files = [os.path.join(gt_root, f + gt_ext) for f in intersect]

    logger.info(
        "Loaded {} images with semantic segmentation from {}".format(len(input_files), image_root)
    )

    dataset_dicts = []
    for (img_path, gt_path) in zip(input_files, gt_files):
        record = {}
        record["file_name"] = img_path
        record["sem_seg_file_name"] = gt_path
        dataset_dicts.append(record)

    return dataset_dicts


def convert_to_coco_dict(dataset_name):
    """
    Convert an instance detection/segmentation or keypoint detection dataset
    in detectron2's standard format into COCO json format.

    Generic dataset description can be found here:
    https://detectron2.readthedocs.io/tutorials/datasets.html#register-a-dataset

    COCO data format description can be found here:
    http://cocodataset.org/#format-data

    Args:
        dataset_name (str):
            name of the source dataset
            Must be registered in DatastCatalog and in detectron2's standard format.
            Must have corresponding metadata "thing_classes"
    Returns:
        coco_dict: serializable dict in COCO json format
    """

    dataset_dicts = DatasetCatalog.get(dataset_name)
    metadata = MetadataCatalog.get(dataset_name)

    # unmap the category mapping ids for COCO
    if hasattr(metadata, "thing_dataset_id_to_contiguous_id"):
        reverse_id_mapping = {v: k for k, v in metadata.thing_dataset_id_to_contiguous_id.items()}
        reverse_id_mapper = lambda contiguous_id: reverse_id_mapping[contiguous_id]  # noqa
    else:
        reverse_id_mapper = lambda contiguous_id: contiguous_id  # noqa

    categories = [
        {"id": reverse_id_mapper(id), "name": name}
        for id, name in enumerate(metadata.thing_classes)
    ]

    logger.info("Converting dataset dicts into COCO format")
    coco_images = []
    coco_annotations = []

    for image_id, image_dict in enumerate(dataset_dicts):
        coco_image = {
            "id": image_dict.get("image_id", image_id),
            "width": int(image_dict["width"]),
            "height": int(image_dict["height"]),
            "file_name": str(image_dict["file_name"]),
        }
        coco_images.append(coco_image)

        anns_per_image = image_dict.get("annotations", [])
        for annotation in anns_per_image:
            # create a new dict with only COCO fields
            coco_annotation = {}

            # COCO requirement: XYWH box format for axis-align and XYWHA for rotated
            bbox = annotation["bbox"]
            if isinstance(bbox, np.ndarray):
                if bbox.ndim != 1:
                    raise ValueError(f"bbox has to be 1-dimensional. Got shape={bbox.shape}.")
                bbox = bbox.tolist()
            if len(bbox) not in [4, 5]:
                raise ValueError(f"bbox has to has length 4 or 5. Got {bbox}.")
            from_bbox_mode = annotation["bbox_mode"]
            to_bbox_mode = BoxMode.XYWH_ABS if len(bbox) == 4 else BoxMode.XYWHA_ABS
            bbox = BoxMode.convert(bbox, from_bbox_mode, to_bbox_mode)

            # COCO requirement: instance area
            if "segmentation" in annotation:
                # Computing areas for instances by counting the pixels
                segmentation = annotation["segmentation"]
                # TODO: check segmentation type: RLE, BinaryMask or Polygon
                if isinstance(segmentation, list):
                    polygons = PolygonMasks([segmentation])
                    area = polygons.area()[0].item()
                elif isinstance(segmentation, dict):  # RLE
                    area = mask_util.area(segmentation).item()
                else:
                    raise TypeError(f"Unknown segmentation type {type(segmentation)}!")
            else:
                # Computing areas using bounding boxes
                if to_bbox_mode == BoxMode.XYWH_ABS:
                    bbox_xy = BoxMode.convert(bbox, to_bbox_mode, BoxMode.XYXY_ABS)
                    area = Boxes([bbox_xy]).area()[0].item()
                else:
                    area = RotatedBoxes([bbox]).area()[0].item()

            if "keypoints" in annotation:
                keypoints = annotation["keypoints"]  # list[int]
                for idx, v in enumerate(keypoints):
                    if idx % 3 != 2:
                        # COCO's segmentation coordinates are floating points in [0, H or W],
                        # but keypoint coordinates are integers in [0, H-1 or W-1]
                        # For COCO format consistency we substract 0.5
                        # https://github.com/facebookresearch/detectron2/pull/175#issuecomment-551202163
                        keypoints[idx] = v - 0.5
                if "num_keypoints" in annotation:
                    num_keypoints = annotation["num_keypoints"]
                else:
                    num_keypoints = sum(kp > 0 for kp in keypoints[2::3])

            # COCO requirement:
            #   linking annotations to images
            #   "id" field must start with 1
            coco_annotation["id"] = len(coco_annotations) + 1
            coco_annotation["image_id"] = coco_image["id"]
            coco_annotation["bbox"] = [round(float(x), 3) for x in bbox]
            coco_annotation["area"] = float(area)
            coco_annotation["iscrowd"] = int(annotation.get("iscrowd", 0))
            coco_annotation["category_id"] = int(reverse_id_mapper(annotation["category_id"]))

            # Add optional fields
            if "keypoints" in annotation:
                coco_annotation["keypoints"] = keypoints
                coco_annotation["num_keypoints"] = num_keypoints

            if "segmentation" in annotation:
                seg = coco_annotation["segmentation"] = annotation["segmentation"]
                if isinstance(seg, dict):  # RLE
                    counts = seg["counts"]
                    if not isinstance(counts, str):
                        # make it json-serializable
                        seg["counts"] = counts.decode("ascii")

            coco_annotations.append(coco_annotation)

    logger.info(
        "Conversion finished, "
        f"#images: {len(coco_images)}, #annotations: {len(coco_annotations)}"
    )

    info = {
        "date_created": str(datetime.datetime.now()),
        "description": "Automatically generated COCO json file for Detectron2.",
    }
    coco_dict = {"info": info, "images": coco_images, "categories": categories, "licenses": None}
    if len(coco_annotations) > 0:
        coco_dict["annotations"] = coco_annotations
    return coco_dict


def convert_to_coco_json(dataset_name, output_file, allow_cached=True):
    """
    Converts dataset into COCO format and saves it to a json file.
    dataset_name must be registered in DatasetCatalog and in detectron2's standard format.

    Args:
        dataset_name:
            reference from the config file to the catalogs
            must be registered in DatasetCatalog and in detectron2's standard format
        output_file: path of json file that will be saved to
        allow_cached: if json file is already present then skip conversion
    """

    # TODO: The dataset or the conversion script *may* change,
    # a checksum would be useful for validating the cached data

    PathManager.mkdirs(os.path.dirname(output_file))
    with file_lock(output_file):
        if PathManager.exists(output_file) and allow_cached:
            logger.warning(
                f"Using previously cached COCO format annotations at '{output_file}'. "
                "You need to clear the cache file if your dataset has been modified."
            )
        else:
            logger.info(f"Converting annotations of dataset '{dataset_name}' to COCO format ...)")
            coco_dict = convert_to_coco_dict(dataset_name)

            logger.info(f"Caching COCO format annotations at '{output_file}' ...")
            tmp_file = output_file + ".tmp"
            with PathManager.open(tmp_file, "w") as f:
                json.dump(coco_dict, f)
            shutil.move(tmp_file, output_file)


def register_coco_instances(name, metadata, json_file, image_root):
    """
    Register a dataset in COCO's json annotation format for
    instance detection, instance segmentation and keypoint detection.
    (i.e., Type 1 and 2 in http://cocodataset.org/#format-data.
    `instances*.json` and `person_keypoints*.json` in the dataset).

    This is an example of how to register a new dataset.
    You can do something similar to this function, to register new datasets.

    Args:
        name (str): the name that identifies a dataset, e.g. "coco_2014_train".
        metadata (dict): extra metadata associated with this dataset.  You can
            leave it as an empty dict.
        json_file (str): path to the json instance annotation file.
        image_root (str or path-like): directory which contains all the images.
    """
    assert isinstance(name, str), name
    assert isinstance(json_file, (str, os.PathLike)), json_file
    assert isinstance(image_root, (str, os.PathLike)), image_root
    # 1. register a function which returns dicts
    DatasetCatalog.register(name, lambda: load_coco_json(json_file, image_root, name))

    # 2. Optionally, add metadata about this dataset,
    # since they might be useful in evaluation, visualization or logging
    MetadataCatalog.get(name).set(
        json_file=json_file, image_root=image_root, evaluator_type="coco", **metadata
    )


if __name__ == "__main__":
    """
    Test the COCO json dataset loader.

    Usage:
        python -m detectron2.data.datasets.coco \
            path/to/json path/to/image_root dataset_name

        "dataset_name" can be "coco_2014_minival_100", or other
        pre-registered ones
    """
    from logger import setup_logger
    from visualizer import Visualizer
    import datasets  # noqa # add pre-defined metadata
    import sys

    logger = setup_logger(name=__name__)
    assert sys.argv[3] in DatasetCatalog.list()
    meta = MetadataCatalog.get(sys.argv[3])

    dicts = load_coco_json(sys.argv[1], sys.argv[2], sys.argv[3])
    logger.info("Done loading {} samples.".format(len(dicts)))

    dirname = "coco-data-vis"
    os.makedirs(dirname, exist_ok=True)
    for d in dicts:
        img = np.array(Image.open(d["file_name"]))
        visualizer = Visualizer(img, metadata=meta)
        vis = visualizer.draw_dataset_dict(d)
        fpath = os.path.join(dirname, os.path.basename(d["file_name"]))
        vis.save(fpath)


class RotatedBoxes(Boxes):
    """
    This structure stores a list of rotated boxes as a Nx5 torch.Tensor.
    It supports some common methods about boxes
    (`area`, `clip`, `nonempty`, etc),
    and also behaves like a Tensor
    (support indexing, `to(device)`, `.device`, and iteration over all boxes)
    """

    def __init__(self, tensor: torch.Tensor):
        """
        Args:
            tensor (Tensor[float]): a Nx5 matrix.  Each row is
                (x_center, y_center, width, height, angle),
                in which angle is represented in degrees.
                While there's no strict range restriction for it,
                the recommended principal range is between [-180, 180) degrees.

        Assume we have a horizontal box B = (x_center, y_center, width, height),
        where width is along the x-axis and height is along the y-axis.
        The rotated box B_rot (x_center, y_center, width, height, angle)
        can be seen as:

        1. When angle == 0:
           B_rot == B
        2. When angle > 0:
           B_rot is obtained by rotating B w.r.t its center by :math:`|angle|` degrees CCW;
        3. When angle < 0:
           B_rot is obtained by rotating B w.r.t its center by :math:`|angle|` degrees CW.

        Mathematically, since the right-handed coordinate system for image space
        is (y, x), where y is top->down and x is left->right, the 4 vertices of the
        rotated rectangle :math:`(yr_i, xr_i)` (i = 1, 2, 3, 4) can be obtained from
        the vertices of the horizontal rectangle :math:`(y_i, x_i)` (i = 1, 2, 3, 4)
        in the following way (:math:`\\theta = angle*\\pi/180` is the angle in radians,
        :math:`(y_c, x_c)` is the center of the rectangle):

        .. math::

            yr_i = \\cos(\\theta) (y_i - y_c) - \\sin(\\theta) (x_i - x_c) + y_c,

            xr_i = \\sin(\\theta) (y_i - y_c) + \\cos(\\theta) (x_i - x_c) + x_c,

        which is the standard rigid-body rotation transformation.

        Intuitively, the angle is
        (1) the rotation angle from y-axis in image space
        to the height vector (top->down in the box's local coordinate system)
        of the box in CCW, and
        (2) the rotation angle from x-axis in image space
        to the width vector (left->right in the box's local coordinate system)
        of the box in CCW.

        More intuitively, consider the following horizontal box ABCD represented
        in (x1, y1, x2, y2): (3, 2, 7, 4),
        covering the [3, 7] x [2, 4] region of the continuous coordinate system
        which looks like this:

        .. code:: none

            O--------> x
            |
            |  A---B
            |  |   |
            |  D---C
            |
            v y

        Note that each capital letter represents one 0-dimensional geometric point
        instead of a 'square pixel' here.

        In the example above, using (x, y) to represent a point we have:

        .. math::

            O = (0, 0), A = (3, 2), B = (7, 2), C = (7, 4), D = (3, 4)

        We name vector AB = vector DC as the width vector in box's local coordinate system, and
        vector AD = vector BC as the height vector in box's local coordinate system. Initially,
        when angle = 0 degree, they're aligned with the positive directions of x-axis and y-axis
        in the image space, respectively.

        For better illustration, we denote the center of the box as E,

        .. code:: none

            O--------> x
            |
            |  A---B
            |  | E |
            |  D---C
            |
            v y

        where the center E = ((3+7)/2, (2+4)/2) = (5, 3).

        Also,

        .. math::

            width = |AB| = |CD| = 7 - 3 = 4,
            height = |AD| = |BC| = 4 - 2 = 2.

        Therefore, the corresponding representation for the same shape in rotated box in
        (x_center, y_center, width, height, angle) format is:

        (5, 3, 4, 2, 0),

        Now, let's consider (5, 3, 4, 2, 90), which is rotated by 90 degrees
        CCW (counter-clockwise) by definition. It looks like this:

        .. code:: none

            O--------> x
            |   B-C
            |   | |
            |   |E|
            |   | |
            |   A-D
            v y

        The center E is still located at the same point (5, 3), while the vertices
        ABCD are rotated by 90 degrees CCW with regard to E:
        A = (4, 5), B = (4, 1), C = (6, 1), D = (6, 5)

        Here, 90 degrees can be seen as the CCW angle to rotate from y-axis to
        vector AD or vector BC (the top->down height vector in box's local coordinate system),
        or the CCW angle to rotate from x-axis to vector AB or vector DC (the left->right
        width vector in box's local coordinate system).

        .. math::

            width = |AB| = |CD| = 5 - 1 = 4,
            height = |AD| = |BC| = 6 - 4 = 2.

        Next, how about (5, 3, 4, 2, -90), which is rotated by 90 degrees CW (clockwise)
        by definition? It looks like this:

        .. code:: none

            O--------> x
            |   D-A
            |   | |
            |   |E|
            |   | |
            |   C-B
            v y

        The center E is still located at the same point (5, 3), while the vertices
        ABCD are rotated by 90 degrees CW with regard to E:
        A = (6, 1), B = (6, 5), C = (4, 5), D = (4, 1)

        .. math::

            width = |AB| = |CD| = 5 - 1 = 4,
            height = |AD| = |BC| = 6 - 4 = 2.

        This covers exactly the same region as (5, 3, 4, 2, 90) does, and their IoU
        will be 1. However, these two will generate different RoI Pooling results and
        should not be treated as an identical box.

        On the other hand, it's easy to see that (X, Y, W, H, A) is identical to
        (X, Y, W, H, A+360N), for any integer N. For example (5, 3, 4, 2, 270) would be
        identical to (5, 3, 4, 2, -90), because rotating the shape 270 degrees CCW is
        equivalent to rotating the same shape 90 degrees CW.

        We could rotate further to get (5, 3, 4, 2, 180), or (5, 3, 4, 2, -180):

        .. code:: none

            O--------> x
            |
            |  C---D
            |  | E |
            |  B---A
            |
            v y

        .. math::

            A = (7, 4), B = (3, 4), C = (3, 2), D = (7, 2),

            width = |AB| = |CD| = 7 - 3 = 4,
            height = |AD| = |BC| = 4 - 2 = 2.

        Finally, this is a very inaccurate (heavily quantized) illustration of
        how (5, 3, 4, 2, 60) looks like in case anyone wonders:

        .. code:: none

            O--------> x
            |     B\
            |    /  C
            |   /E /
            |  A  /
            |   `D
            v y

        It's still a rectangle with center of (5, 3), width of 4 and height of 2,
        but its angle (and thus orientation) is somewhere between
        (5, 3, 4, 2, 0) and (5, 3, 4, 2, 90).
        """
        device = tensor.device if isinstance(tensor, torch.Tensor) else torch.device("cpu")
        tensor = torch.as_tensor(tensor, dtype=torch.float32, device=device)
        if tensor.numel() == 0:
            # Use reshape, so we don't end up creating a new tensor that does not depend on
            # the inputs (and consequently confuses jit)
            tensor = tensor.reshape((0, 5)).to(dtype=torch.float32, device=device)
        assert tensor.dim() == 2 and tensor.size(-1) == 5, tensor.size()

        self.tensor = tensor

    def clone(self) -> "RotatedBoxes":
        """
        Clone the RotatedBoxes.

        Returns:
            RotatedBoxes
        """
        return RotatedBoxes(self.tensor.clone())

    @_maybe_jit_unused
    def to(self, device: torch.device):
        # Boxes are assumed float32 and does not support to(dtype)
        return RotatedBoxes(self.tensor.to(device=device))

    def area(self) -> torch.Tensor:
        """
        Computes the area of all the boxes.

        Returns:
            torch.Tensor: a vector with areas of each box.
        """
        box = self.tensor
        area = box[:, 2] * box[:, 3]
        return area

    def normalize_angles(self) -> None:
        """
        Restrict angles to the range of [-180, 180) degrees
        """
        self.tensor[:, 4] = (self.tensor[:, 4] + 180.0) % 360.0 - 180.0

    def clip(self, box_size: Tuple[int, int], clip_angle_threshold: float = 1.0) -> None:
        """
        Clip (in place) the boxes by limiting x coordinates to the range [0, width]
        and y coordinates to the range [0, height].

        For RRPN:
        Only clip boxes that are almost horizontal with a tolerance of
        clip_angle_threshold to maintain backward compatibility.

        Rotated boxes beyond this threshold are not clipped for two reasons:

        1. There are potentially multiple ways to clip a rotated box to make it
           fit within the image.
        2. It's tricky to make the entire rectangular box fit within the image
           and still be able to not leave out pixels of interest.

        Therefore we rely on ops like RoIAlignRotated to safely handle this.

        Args:
            box_size (height, width): The clipping box's size.
            clip_angle_threshold:
                Iff. abs(normalized(angle)) <= clip_angle_threshold (in degrees),
                we do the clipping as horizontal boxes.
        """
        h, w = box_size

        # normalize angles to be within (-180, 180] degrees
        self.normalize_angles()

        idx = torch.where(torch.abs(self.tensor[:, 4]) <= clip_angle_threshold)[0]

        # convert to (x1, y1, x2, y2)
        x1 = self.tensor[idx, 0] - self.tensor[idx, 2] / 2.0
        y1 = self.tensor[idx, 1] - self.tensor[idx, 3] / 2.0
        x2 = self.tensor[idx, 0] + self.tensor[idx, 2] / 2.0
        y2 = self.tensor[idx, 1] + self.tensor[idx, 3] / 2.0

        # clip
        x1.clamp_(min=0, max=w)
        y1.clamp_(min=0, max=h)
        x2.clamp_(min=0, max=w)
        y2.clamp_(min=0, max=h)

        # convert back to (xc, yc, w, h)
        self.tensor[idx, 0] = (x1 + x2) / 2.0
        self.tensor[idx, 1] = (y1 + y2) / 2.0
        # make sure widths and heights do not increase due to numerical errors
        self.tensor[idx, 2] = torch.min(self.tensor[idx, 2], x2 - x1)
        self.tensor[idx, 3] = torch.min(self.tensor[idx, 3], y2 - y1)

    def nonempty(self, threshold: float = 0.0) -> torch.Tensor:
        """
        Find boxes that are non-empty.
        A box is considered empty, if either of its side is no larger than threshold.

        Returns:
            Tensor: a binary vector which represents
            whether each box is empty (False) or non-empty (True).
        """
        box = self.tensor
        widths = box[:, 2]
        heights = box[:, 3]
        keep = (widths > threshold) & (heights > threshold)
        return keep

    def __getitem__(self, item) -> "RotatedBoxes":
        """
        Returns:
            RotatedBoxes: Create a new :class:`RotatedBoxes` by indexing.

        The following usage are allowed:

        1. `new_boxes = boxes[3]`: return a `RotatedBoxes` which contains only one box.
        2. `new_boxes = boxes[2:10]`: return a slice of boxes.
        3. `new_boxes = boxes[vector]`, where vector is a torch.ByteTensor
           with `length = len(boxes)`. Nonzero elements in the vector will be selected.

        Note that the returned RotatedBoxes might share storage with this RotatedBoxes,
        subject to Pytorch's indexing semantics.
        """
        if isinstance(item, int):
            return RotatedBoxes(self.tensor[item].view(1, -1))
        b = self.tensor[item]
        assert b.dim() == 2, "Indexing on RotatedBoxes with {} failed to return a matrix!".format(
            item
        )
        return RotatedBoxes(b)

    def __len__(self) -> int:
        return self.tensor.shape[0]

    def __repr__(self) -> str:
        return "RotatedBoxes(" + str(self.tensor) + ")"

    def inside_box(self, box_size: Tuple[int, int], boundary_threshold: int = 0) -> torch.Tensor:
        """
        Args:
            box_size (height, width): Size of the reference box covering
                [0, width] x [0, height]
            boundary_threshold (int): Boxes that extend beyond the reference box
                boundary by more than boundary_threshold are considered "outside".

        For RRPN, it might not be necessary to call this function since it's common
        for rotated box to extend to outside of the image boundaries
        (the clip function only clips the near-horizontal boxes)

        Returns:
            a binary vector, indicating whether each box is inside the reference box.
        """
        height, width = box_size

        cnt_x = self.tensor[..., 0]
        cnt_y = self.tensor[..., 1]
        half_w = self.tensor[..., 2] / 2.0
        half_h = self.tensor[..., 3] / 2.0
        a = self.tensor[..., 4]
        c = torch.abs(torch.cos(a * math.pi / 180.0))
        s = torch.abs(torch.sin(a * math.pi / 180.0))
        # This basically computes the horizontal bounding rectangle of the rotated box
        max_rect_dx = c * half_w + s * half_h
        max_rect_dy = c * half_h + s * half_w

        inds_inside = (
            (cnt_x - max_rect_dx >= -boundary_threshold)
            & (cnt_y - max_rect_dy >= -boundary_threshold)
            & (cnt_x + max_rect_dx < width + boundary_threshold)
            & (cnt_y + max_rect_dy < height + boundary_threshold)
        )

        return inds_inside

    def get_centers(self) -> torch.Tensor:
        """
        Returns:
            The box centers in a Nx2 array of (x, y).
        """
        return self.tensor[:, :2]

    def scale(self, scale_x: float, scale_y: float) -> None:
        """
        Scale the rotated box with horizontal and vertical scaling factors
        Note: when scale_factor_x != scale_factor_y,
        the rotated box does not preserve the rectangular shape when the angle
        is not a multiple of 90 degrees under resize transformation.
        Instead, the shape is a parallelogram (that has skew)
        Here we make an approximation by fitting a rotated rectangle to the parallelogram.
        """
        self.tensor[:, 0] *= scale_x
        self.tensor[:, 1] *= scale_y
        theta = self.tensor[:, 4] * math.pi / 180.0
        c = torch.cos(theta)
        s = torch.sin(theta)

        # In image space, y is top->down and x is left->right
        # Consider the local coordintate system for the rotated box,
        # where the box center is located at (0, 0), and the four vertices ABCD are
        # A(-w / 2, -h / 2), B(w / 2, -h / 2), C(w / 2, h / 2), D(-w / 2, h / 2)
        # the midpoint of the left edge AD of the rotated box E is:
        # E = (A+D)/2 = (-w / 2, 0)
        # the midpoint of the top edge AB of the rotated box F is:
        # F(0, -h / 2)
        # To get the old coordinates in the global system, apply the rotation transformation
        # (Note: the right-handed coordinate system for image space is yOx):
        # (old_x, old_y) = (s * y + c * x, c * y - s * x)
        # E(old) = (s * 0 + c * (-w/2), c * 0 - s * (-w/2)) = (-c * w / 2, s * w / 2)
        # F(old) = (s * (-h / 2) + c * 0, c * (-h / 2) - s * 0) = (-s * h / 2, -c * h / 2)
        # After applying the scaling factor (sfx, sfy):
        # E(new) = (-sfx * c * w / 2, sfy * s * w / 2)
        # F(new) = (-sfx * s * h / 2, -sfy * c * h / 2)
        # The new width after scaling tranformation becomes:

        # w(new) = |E(new) - O| * 2
        #        = sqrt[(sfx * c * w / 2)^2 + (sfy * s * w / 2)^2] * 2
        #        = sqrt[(sfx * c)^2 + (sfy * s)^2] * w
        # i.e., scale_factor_w = sqrt[(sfx * c)^2 + (sfy * s)^2]
        #
        # For example,
        # when angle = 0 or 180, |c| = 1, s = 0, scale_factor_w == scale_factor_x;
        # when |angle| = 90, c = 0, |s| = 1, scale_factor_w == scale_factor_y
        self.tensor[:, 2] *= torch.sqrt((scale_x * c) ** 2 + (scale_y * s) ** 2)

        # h(new) = |F(new) - O| * 2
        #        = sqrt[(sfx * s * h / 2)^2 + (sfy * c * h / 2)^2] * 2
        #        = sqrt[(sfx * s)^2 + (sfy * c)^2] * h
        # i.e., scale_factor_h = sqrt[(sfx * s)^2 + (sfy * c)^2]
        #
        # For example,
        # when angle = 0 or 180, |c| = 1, s = 0, scale_factor_h == scale_factor_y;
        # when |angle| = 90, c = 0, |s| = 1, scale_factor_h == scale_factor_x
        self.tensor[:, 3] *= torch.sqrt((scale_x * s) ** 2 + (scale_y * c) ** 2)

        # The angle is the rotation angle from y-axis in image space to the height
        # vector (top->down in the box's local coordinate system) of the box in CCW.
        #
        # angle(new) = angle_yOx(O - F(new))
        #            = angle_yOx( (sfx * s * h / 2, sfy * c * h / 2) )
        #            = atan2(sfx * s * h / 2, sfy * c * h / 2)
        #            = atan2(sfx * s, sfy * c)
        #
        # For example,
        # when sfx == sfy, angle(new) == atan2(s, c) == angle(old)
        self.tensor[:, 4] = torch.atan2(scale_x * s, scale_y * c) * 180 / math.pi

    @classmethod
    @_maybe_jit_unused
    def cat(cls, boxes_list: List["RotatedBoxes"]) -> "RotatedBoxes":
        """
        Concatenates a list of RotatedBoxes into a single RotatedBoxes

        Arguments:
            boxes_list (list[RotatedBoxes])

        Returns:
            RotatedBoxes: the concatenated RotatedBoxes
        """
        assert isinstance(boxes_list, (list, tuple))
        if len(boxes_list) == 0:
            return cls(torch.empty(0))
        assert all([isinstance(box, RotatedBoxes) for box in boxes_list])

        # use torch.cat (v.s. layers.cat) so the returned boxes never share storage with input
        cat_boxes = cls(torch.cat([b.tensor for b in boxes_list], dim=0))
        return cat_boxes

    @property
    def device(self) -> torch.device:
        return self.tensor.device

    @torch.jit.unused
    def __iter__(self):
        """
        Yield a box as a Tensor of shape (5,) at a time.
        """
        yield from self.tensor


def pairwise_iou(boxes1: RotatedBoxes, boxes2: RotatedBoxes) -> None:
    """
    Given two lists of rotated boxes of size N and M,
    compute the IoU (intersection over union)
    between **all** N x M pairs of boxes.
    The box order must be (x_center, y_center, width, height, angle).

    Args:
        boxes1, boxes2 (RotatedBoxes):
            two `RotatedBoxes`. Contains N & M rotated boxes, respectively.

    Returns:
        Tensor: IoU, sized [N,M].
    """

    return pairwise_iou_rotated(boxes1.tensor, boxes2.tensor)