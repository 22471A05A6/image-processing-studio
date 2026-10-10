
import cv2
import numpy as np


def apply_blur(image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """Apply basic averaging blur to an image."""

    if image is None or image.size == 0:
        raise ValueError("Input image is empty.")

    if kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError("Kernel size must be a positive odd number.")

    blurred_image = cv2.blur(
        image,
        (kernel_size, kernel_size)
    )

    return blurred_image


def apply_gaussian_blur(
    image: np.ndarray,
    kernel_size: int = 5
) -> np.ndarray:
    """Apply Gaussian blur to an image."""

    if image is None or image.size == 0:
        raise ValueError("Input image is empty.")

    if kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError("Kernel size must be a positive odd number.")

    blurred_image = cv2.GaussianBlur(
        image,
        (kernel_size, kernel_size),
        0
    )

    return blurred_image

def apply_median_blur(
    image: np.ndarray,
    kernel_size: int = 5
) -> np.ndarray:
    """Apply median blur to an image."""

    if image is None or image.size == 0:
        raise ValueError("Input image is empty.")

    if kernel_size < 3 or kernel_size % 2 == 0:
        raise ValueError(
            "Kernel size must be an odd number greater than or equal to 3."
        )

    blurred_image = cv2.medianBlur(
        image,
        kernel_size
    )

    return blurred_image




