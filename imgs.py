import cv2
import numpy as np


def read_rgb_image(img_path):
    return cv2.imread(img_path)[..., ::-1]


def write_rgb_image(img_path: str, img: np.ndarray):
    """Write an RGB image to disk., expected shape is (H, W, 3), RGB format"""
    cv2.imwrite(img_path, img[..., ::-1])


def write_bgr_image(img_path: str, img: np.ndarray):
    """Write a BGR image to disk., expected shape is (H, W, 3), BGR format"""
    cv2.imwrite(img_path, img)
