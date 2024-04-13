from typing import Tuple, Union, List
from torch import Tensor

import numpy as np

import torch

def center_to_corners_torch(bboxes_center: torch.Tensor):
    x_c, y_c, w, h = bboxes_center.unbind(-1)
    b = [
        (x_c - 0.5 * w), (y_c - 0.5 * h),
        (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)

def corner_to_centers_torch(bboxes_corners: torch.Tensor):
    x0, y0, x1, y1 = bboxes_corners.unbind(-1)
    b = [
        (x0 + x1) / 2, (y0 + y1) / 2,
        (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)

def center_to_corners(bboxes_center: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    """
    Converts bounding boxes from center format to corners format.

    center format: contains the coordinate for the center of the box and its width, height dimensions
        (center_x, center_y, width, height)
    corners format: contains the coodinates for the top-left and bottom-right corners of the box
        (top_left_x, top_left_y, bottom_right_x, bottom_right_y)
    """
    if isinstance(bboxes_center, torch.Tensor):
        return center_to_corners_torch(bboxes_center)
    
    center_x, center_y, width, height = bboxes_center.T
    bboxes_corners = np.stack(
        # top left x, top left y, bottom right x, bottom right y
        [center_x - 0.5 * width, center_y - 0.5 * height, center_x + 0.5 * width, center_y + 0.5 * height],
        axis=-1,
    )
    return bboxes_corners

def corner_to_centers(bboxes_corners: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    """
    Converts bounding boxes from corners format to center format.

    corners format: contains the coordinates for the top-left and bottom-right corners of the box
        (top_left_x, top_left_y, bottom_right_x, bottom_right_y)
    center format: contains the coordinate for the center of the box and its the width, height dimensions
        (center_x, center_y, width, height)
    """
    if isinstance(bboxes_corners, torch.Tensor):
        return corner_to_centers_torch(bboxes_corners)
    
    top_left_x, top_left_y, bottom_right_x, bottom_right_y = bboxes_corners.T
    bboxes_center = np.stack(
        [
            (top_left_x + bottom_right_x) / 2,  # center x
            (top_left_y + bottom_right_y) / 2,  # center y
            (bottom_right_x - top_left_x),  # width
            (bottom_right_y - top_left_y),  # height
        ],
        axis=-1,
    )
    return bboxes_center

def scale_bboxes(bboxes_corners: np.ndarray, img_size: Tuple[int, int]) -> np.array:
    """
    scale the bboxes to the image size.
    Args:
        bboxes_corners (`np.ndarray`): bounding box corners in [0,1] scale
        img_size (`Tuple[int, int]`): image width x height
    """
    scale_fct = np.array([img_size[0], img_size[1], img_size[0],img_size[1]])
    scaled_bboxes = bboxes_corners * scale_fct
    scaled_bboxes = scaled_bboxes.astype(np.int32)
    return scaled_bboxes

def _clone(v: Union[Tensor, np.ndarray]) -> Tensor | np.ndarray:
    return v.detach().clone().to(torch.float32) \
        if isinstance(v, Tensor) \
        else np.copy(v).astype(dtype=np.float32)

def convert_cxy_xy_and_scale(
    bboxes_cxy: Union[Tensor, np.ndarray], 
    img_size: Tuple[int, int]) -> np.ndarray | Tensor:
    """
    converts cxy to xy and then scales with the image size
    Args:
        bboxes_cxy (`Tensor` | 'np.ndarray`): normalized bboxes with center, w & h.
        img_size: width, height 
    """
    bboxes_cxy = _clone(bboxes_cxy)
    bboxes_xy = center_to_corners(bboxes_cxy)
    bboxes_xy[:,[0, 2]] = bboxes_xy[:,[0, 2]] * img_size[0]
    bboxes_xy[:,[1, 3]] = bboxes_xy[:,[1, 3]] * img_size[1]
    
    return bboxes_xy

def convert_xy_cxy_and_unscale(
    bboxes_xy: Union[Tensor, np.ndarray], 
    img_size: Tuple[int, int]) -> np.ndarray | Tensor:
    """
    converts xy to cxy and then normalizes with the image size
    Args:
        bboxes_xy (`Tensor` | 'np.ndarray`): bboxes with corners in img size.
        img_size: width, height to normalize the xy
    """
    
    bboxes_xy = _clone(bboxes_xy)
    bboxes_cxy = corner_to_centers(bboxes_xy)
    bboxes_cxy[:,[0, 2]] = bboxes_cxy[:,[0, 2]] / img_size[0]
    bboxes_cxy[:,[1, 3]] = bboxes_cxy[:,[1, 3]] / img_size[1]
    return bboxes_cxy