
import cv2
import numpy as np


def apply_canny(
    image: np.ndarray,
    threshold1: int = 100,
    threshold2: int = 200
) -> np.ndarray:
    """Detect image edges using the Canny algorithm."""

    if image is None or image.size == 0:
        raise ValueError("Input image is empty.")

    if not 0 <= threshold1 <= 255:
        raise ValueError("Threshold 1 must be between 0 and 255.")

    if not 0 <= threshold2 <= 255:
        raise ValueError("Threshold 2 must be between 0 and 255.")

    if threshold1 >= threshold2:
        raise ValueError("Threshold 1 must be smaller than Threshold 2.")

    gray_image = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2GRAY
    )

    edges = cv2.Canny(
        gray_image,
        threshold1,
        threshold2
    )

    return edges

def apply_sobel(image: np.ndarray) -> np.ndarray:
    """Detect image edges using the Sobel operator."""

    if image is None or image.size == 0:
        raise ValueError("Input image is empty.")

    gray_image = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2GRAY
    )

    # Detect horizontal intensity changes
    sobel_x = cv2.Sobel(
        gray_image,
        cv2.CV_64F,
        1,
        0,
        ksize=3
    )

    # Detect vertical intensity changes
    sobel_y = cv2.Sobel(
        gray_image,
        cv2.CV_64F,
        0,
        1,
        ksize=3
    )

    # Combine both gradient directions
    magnitude = cv2.magnitude(
        sobel_x,
        sobel_y
    )

    # Convert output to displayable 8-bit image
    sobel_image = np.clip(
        magnitude,
        0,
        255
    ).astype(np.uint8)

    return sobel_image
