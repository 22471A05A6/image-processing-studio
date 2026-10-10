
import cv2
import numpy as np


def convert_to_grayscale(image: np.ndarray) -> np.ndarray:
    """
    Convert an RGB image into a grayscale image.

    Args:
        image: RGB image as a NumPy array.

    Returns:
        Grayscale image as a NumPy array.
    """
    if image is None or image.size == 0:
        raise ValueError("Input image is empty.")

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected an RGB image with 3 channels.")

    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
