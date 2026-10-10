
import cv2
import numpy as np


def apply_threshold(
    image: np.ndarray,
    threshold_value: int = 127,
    max_value: int = 255
) -> np.ndarray:
    """Apply binary thresholding to an image."""

    if image is None or image.size == 0:
        raise ValueError("Input image is empty.")

    if not 0 <= threshold_value <= 255:
        raise ValueError("Threshold value must be between 0 and 255.")

    if not 1 <= max_value <= 255:
        raise ValueError("Maximum value must be between 1 and 255.")

    gray_image = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2GRAY
    )

    _, threshold_image = cv2.threshold(
        gray_image,
        threshold_value,
        max_value,
        cv2.THRESH_BINARY
    )

    return threshold_image

def apply_adaptive_threshold(
    image: np.ndarray,
    max_value: int = 255,
    block_size: int = 11,
    c_value: int = 2
) -> np.ndarray:
    """Apply adaptive Gaussian thresholding to an image."""

    if image is None or image.size == 0:
        raise ValueError("Input image is empty.")

    if not 1 <= max_value <= 255:
        raise ValueError("Maximum value must be between 1 and 255.")

    if block_size < 3 or block_size % 2 == 0:
        raise ValueError(
            "Block size must be an odd number greater than or equal to 3."
        )

    gray_image = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2GRAY
    )

    adaptive_image = cv2.adaptiveThreshold(
        gray_image,
        max_value,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size,
        c_value
    )

    return adaptive_image
