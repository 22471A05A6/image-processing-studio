
import numpy as np


def adjust_brightness(
    image: np.ndarray,
    brightness: int
) -> np.ndarray:
    """
    Adjust brightness from -100 to +100.

    Negative values darken the image.
    Positive values brighten the image.
    """
    if image is None or image.size == 0:
        raise ValueError("Input image is empty.")

    if not -100 <= brightness <= 100:
        raise ValueError("Brightness must be between -100 and 100.")

    adjusted = image.astype(np.int16) + brightness

    return np.clip(adjusted, 0, 255).astype(np.uint8)

def adjust_contrast(
    image: np.ndarray,
    factor: float
) -> np.ndarray:
    """Adjust image contrast using a midpoint of 127.5."""

    if image is None or image.size == 0:
        raise ValueError("Input image is empty.")

    if not 0.0 <= factor <= 3.0:
        raise ValueError("Contrast factor must be between 0 and 3.")

    adjusted = (
        (image.astype(np.float32) - 127.5) * factor
        + 127.5
    )

    return np.clip(adjusted, 0, 255).astype(np.uint8)

def invert_colors(image: np.ndarray) -> np.ndarray:
    """Invert the colors of an image."""

    if image is None or image.size == 0:
        raise ValueError("Input image is empty.")

    inverted_image = 255 - image

    return inverted_image

