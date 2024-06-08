import numpy as np
from matplotlib import pyplot as plt
import cv2


def resize_longest_side(image, target_length):
    long_side_length = target_length
    oldh, oldw = image.shape[0], image.shape[1]
    scale = long_side_length * 1.0 / max(oldh, oldw)
    newh, neww = int(oldh * scale + 0.5), int(oldw * scale + 0.5)
    return cv2.resize(image, (neww, newh), interpolation=cv2.INTER_AREA)

def pad_image(image, target_size):
    h, w = image.shape[0], image.shape[1]
    padh = target_size - h
    padw = target_size - w
    if len(image.shape) == 3:
        return np.pad(image, ((0, padh), (0, padw), (0, 0)))
    else:
        return np.pad(image, ((0, padh), (0, padw)))

def get_bbox(gt2D, bbox_shift=5):
    y_indices, x_indices = np.where(gt2D > 0)
    x_min, x_max = x_indices.min(), x_indices.max()
    y_min, y_max = y_indices.min(), y_indices.max()
    return np.array([x_min - bbox_shift, y_min - bbox_shift, 
                     x_max + bbox_shift + 1, y_max + bbox_shift + 1])

def resize_box(box: np.ndarray, 
               new_size: tuple, 
               original_size: tuple) -> np.ndarray:    
    new_box = np.zeros_like(box)
    scale = max(original_size) / max(new_size)
    for i in range(len(box)):
       new_box[i] = int(box[i] * scale)
    return new_box

def show_mask(mask, ax, mask_color=None, alpha=0.5):
    if mask_color is not None:
        color = np.concatenate([mask_color, np.array([alpha])], axis=0)
    else:
        color = np.array([251 / 255, 252 / 255, 30 / 255, alpha])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_box(box, ax, edgecolor=(0, 1, 0)):
    x0, y0, x1, y1 = box
    ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, edgecolor=edgecolor, facecolor=(0, 0, 0, 0), lw=2))
