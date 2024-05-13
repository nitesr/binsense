from typing import Tuple, Union, List
from torch import Tensor

from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks
from torchvision.tv_tensors import Mask
from torchvision.ops import box_convert
import torchvision.transforms.v2  as transforms
from PIL.Image import Image as PILImage

import numpy as np
import torch, PIL


def create_polygon_mask(
        image_size: Tuple[int, int], 
        vertices: Union[np.ndarray, List[Tuple[float, float]]]):
    """
    Create a grayscale image with a white polygonal area on a black background.

    Parameters:
    - image_size (tuple): A tuple representing the dimensions (width, height) of the image.
    - vertices (list): A list of tuples, each containing the x, y coordinates of a vertex
                        of the polygon. Vertices should be in clockwise or counter-clockwise order.

    Returns:
    - PIL.Image.Image: A PIL Image object containing the polygonal mask.
    """

    # Create a new black image with the given dimensions
    mask_img = PIL.Image.new('L', image_size, 0)

    # Draw the polygon on the image. The area inside the polygon will be white (255).
    PIL.ImageDraw.Draw(mask_img, 'L').polygon(vertices, fill=(255))
    
    return mask_img


def annotate_image(
        image: PILImage, 
        labels: List[str], 
        bboxes_cxy: np.ndarray, 
        seg_coords: List[np.array],
        colors: List[Tuple[int, int, int]] = None,
        normalize: bool = True,
        convert_cxy_xy: bool = True) -> PILImage:
    """
    Annotate image with segmentation mask
    """

    if labels is not None and bboxes_cxy is not None and len(bboxes_cxy) != len(labels):
        raise ValueError(f"size of bboxes({len(bboxes_cxy)}) and labels({len(labels)}) doesn't match.")

    if labels is not None and seg_coords is not None and len(seg_coords) != len(labels):
        raise ValueError(f"size of segments({len(seg_coords)}) and labels({len(labels)}) doesn't match.")
    
    if bboxes_cxy is None and seg_coords is None:
        raise ValueError('either bbox or seg are required!')
    
    n_annotations = len(bboxes_cxy) if bboxes_cxy is not None else len(seg_coords)
    assert n_annotations > 0

    if colors and len(colors) != n_annotations:
        raise ValueError('size of colors and labels doesn\'t match.')
    
    if not colors:
        colors=[(255,255,0)]*n_annotations
    
    def unnorm_bboxes():
        bb = bboxes_cxy.copy()
        bb[:,[0,2]] *= image.width
        bb[:,[1,3]] *= image.height
        return bb
    
    def unnorm_and_mask():
        mask_imgs = []
        for c in seg_coords:
          cc = c.copy()
          cc[:,0] *= image.width
          cc[:,1] *= image.height
          mask_imgs.append(create_polygon_mask(image.size, cc))
        return mask_imgs  
     
    mask_imgs = ( unnorm_and_mask() if normalize else seg_coords) if seg_coords is not None else []
    bboxes = ( unnorm_bboxes() if normalize else bboxes_cxy ) if bboxes_cxy is not None else []
    
    annotated_tensor = transforms.PILToTensor()(image)
    # Convert mask images to tensors
    if seg_coords is not None:
        masks = torch.concat([Mask(transforms.PILToTensor()(mask_img), dtype=torch.bool) for mask_img in mask_imgs])

        # Annotate the sample image with segmentation masks
        annotated_tensor = draw_segmentation_masks(
            image=annotated_tensor, 
            masks=masks, 
            alpha=0.2,
            colors=colors
        )

    if bboxes_cxy is not None:
        # Annotate the sample image with labels and bounding boxes
        annotated_tensor = draw_bounding_boxes(
            image=annotated_tensor, 
            boxes=box_convert(torch.Tensor(bboxes), 'cxcywh', 'xyxy') if convert_cxy_xy else torch.Tensor(bboxes),
            labels=labels,
            colors=colors
        )

    return PIL.Image.fromarray(np.moveaxis(annotated_tensor.numpy(), 0, -1))


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
