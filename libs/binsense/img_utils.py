import numpy as np
from typing import Tuple

def center_to_corners(bboxes_center: np.ndarray) -> np.ndarray:
    """
    Converts bounding boxes from center format to corners format.

    center format: contains the coordinate for the center of the box and its width, height dimensions
        (center_x, center_y, width, height)
    corners format: contains the coodinates for the top-left and bottom-right corners of the box
        (top_left_x, top_left_y, bottom_right_x, bottom_right_y)
    """
    center_x, center_y, width, height = bboxes_center.T
    bboxes_corners = np.stack(
        # top left x, top left y, bottom right x, bottom right y
        [center_x - 0.5 * width, center_y - 0.5 * height, center_x + 0.5 * width, center_y + 0.5 * height],
        axis=-1,
    )
    return bboxes_corners

def corner_to_centers(bboxes_corners: np.ndarray) -> np.ndarray:
    """
    Converts bounding boxes from corners format to center format.

    corners format: contains the coordinates for the top-left and bottom-right corners of the box
        (top_left_x, top_left_y, bottom_right_x, bottom_right_y)
    center format: contains the coordinate for the center of the box and its the width, height dimensions
        (center_x, center_y, width, height)
    """
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