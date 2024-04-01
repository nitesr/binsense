from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from typing import List, Tuple, Any, Union, Optional
import numpy as np
import cv2 as cv


# draws the bounding boxes on the cv image
# and plots it as one image
def plot_bboxes(
    cv_img: np.ndarray, 
    bboxes: np.ndarray,
    labels: List[str],
    ax: Axes = None,
    title: str=None):
    """
    draws the bounding boxes on the cv image
        and plots it as one image
    Args:
    
    """
    if ax is None:
        ax = plt.gca()
    
    draw_img = cv_img.copy()
    if not bboxes:
        return
    
    label_txts = []
    for i, bbox in enumerate(bboxes):
        bbox = [int(k) for k in bbox]
        draw_img = cv.rectangle(draw_img, bbox[:2], bbox[2:], (0,0,255), 2)
        if labels and labels[i]:
            label_txts.append((bbox[0], bbox[1], labels[i]))
    
    
    ax.imshow(draw_img, cmap = plt.cm.Spectral)
    for x, y, txt in label_txts:
        ax.text(
            x, y, txt,
            ha='left', va='top',
            color='black',
            fontsize=8,
            bbox={
                'facecolor': 'white',
                'alpha': 0,
                'edgecolor': 'lime',
                'boxstyle': 'square,pad=.1',
            }
        )
        
    if title:
        ax.set_title(title)


def show_images_with_bboxes(
    cv_imgs : List[np.ndarray], 
    bboxes: Optional[List[np.ndarray]] = None, 
    bbox_labels: Optional[List[List[str]]] = None,
    bbox_scores: Optional[List[List[float]]] = None,
    grid: Optional[Tuple[int, int]] = (1, 1), 
    title: Optional[str]=None):
    """
    draw images in a grid. optionally with bboxes and labels if passed.
    Args:
        cv_imgs (`List[np.ndarray]`): images to display
        bboxes (`List[np.ndarray]`) = None: 
            bboxes to plot of image. if passed, must match the size of cv_imgs
        bbox_labels (`List[List[str]]`) = None:
            bbox labels. if passed, must match the size of cv_imgs and bboxes
        bbox_scores (`List[List[float]]`) = None: Not supported yet.
        grid (`Tuple[int, int]`) = (1, 1): images grid size, if mre blank grids will be shown.
        title (`str=None`): title of the grid
    """
    if bboxes and len(cv_imgs) != len(bboxes):
        raise ValueError('size of images and bboxes doesn\'t match.')
    
    if bbox_labels and len(cv_imgs) != len(bbox_labels):
        raise ValueError('size of images and labels doesn\'t match.')

    if bbox_scores and len(cv_imgs) != len(bbox_scores):
        raise ValueError('size of images and scores doesn\'t match.')
    
    rows, cols = grid
    fig, axs = plt.subplots(rows, cols, figsize=(cols*3, rows*3))
    if rows == 1:
        axs = [axs]
    
    for i in range(0, rows):
        for j in range(0, cols):
            ii = i*cols+j
            if ii >= len(cv_imgs):
                continue
            
            plot_bboxes(
                cv_imgs[ii], 
                bboxes[ii] if bboxes else None,
                bbox_labels[ii] if bbox_labels else None,
                axs[i][j],
            )
    
    if title:
        fig.suptitle(title)
    plt.show()
    
# draws the bounding boxes on the cv image
# and plots it in the grid with each image+bbox per grid
def show_bbox_ingrid(
    cv_img : np.ndarray, 
    box_scores: List[Union[Tuple, np.array]], 
    grid: Tuple[int, int], 
    title: str=None):
    """
    draws the bounding boxes on the cv image
        and plots it in the grid with each image+bbox per grid
    
    Args:
        cv_img (`np.ndarray`):
            open cv image on which the bbox needs to drawn
        box_scores (`List[Union[Tuple, np.array]]`):
            list of bounding boxes in tuples of box corner (left-top/right-bottom) coordinates and scores
            or list of bounding boxes with no scores
        grid (`Tuple[int, int]`):
            matplotlib plot grid rowsxcols
        title: str=None
            title of the figure
    """
    rows, cols = grid
    fig, axs = plt.subplots(rows, cols, figsize=(cols*3, rows*3))
    if rows == 1:
        axs = [axs]
        
    for i in range(0, rows):
        for j in range(0, cols):
            bi = i*cols+j
            if bi >= len(box_scores):
                continue
            
            bbox, score = box_scores[bi] if isinstance(box_scores[bi], tuple) \
                else (box_scores[bi], None)
            
            plot_bboxes(
                cv_img, 
                np.expand_dims(bbox, axis=0),
                [f'Score: {score:1.3f}' if score else None],
                axs[i][j]
            )
    if title:
        fig.suptitle(title)
    plt.show()
