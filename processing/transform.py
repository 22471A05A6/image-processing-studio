
import cv2
import numpy as np


def resize_image(
    image: np.ndarray,
    width: int,
    height: int
) -> np.ndarray:
    """
    Resize an RGB image to the specified dimensions.
    """
    if image is None or image.size == 0:
        raise ValueError("Input image is empty.")

    if width < 1 or height < 1:
        raise ValueError("Width and height must be positive.")

    resized = cv2.resize(
        image,
        (width, height),
        interpolation=cv2.INTER_AREA
    )

    return resized


def crop_image(
    image: np.ndarray,
    x: int,
    y: int,
    width: int,
    height: int
) -> np.ndarray:
    """Crop an RGB image using x, y, width and height."""

    if image is None or image.size == 0:
        raise ValueError("Input image is empty.")

    image_height, image_width = image.shape[:2]

    if x < 0 or y < 0:
        raise ValueError("X and Y must be non-negative.")

    if width < 1 or height < 1:
        raise ValueError("Crop width and height must be positive.")

    if x + width > image_width or y + height > image_height:
        raise ValueError("Crop region exceeds image boundaries.")

    return image[y:y + height, x:x + width]


def rotate_image(
    image: np.ndarray,
    angle: float
) -> np.ndarray:
    """Rotate an RGB image counterclockwise by the given angle."""

    if image is None or image.size == 0:
        raise ValueError("Input image is empty.")

    height, width = image.shape[:2]
    center = (width / 2, height / 2)

    matrix = cv2.getRotationMatrix2D(
        center,
        angle,
        1.0
    )

    cos_value = abs(matrix[0, 0])
    sin_value = abs(matrix[0, 1])

    new_width = int(height * sin_value + width * cos_value)
    new_height = int(height * cos_value + width * sin_value)

    matrix[0, 2] += new_width / 2 - center[0]
    matrix[1, 2] += new_height / 2 - center[1]

    return cv2.warpAffine(
        image,
        matrix,
        (new_width, new_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0)
    )


def flip_image(
    image: np.ndarray,
    direction: str
) -> np.ndarray:
    """Flip an RGB image horizontally, vertically, or both."""

    if image is None or image.size == 0:
        raise ValueError("Input image is empty.")

    if direction == "Horizontal":
        return cv2.flip(image, 1)

    elif direction == "Vertical":
        return cv2.flip(image, 0)

    elif direction == "Both":
        return cv2.flip(image, -1)

    else:
        raise ValueError(
            "Direction must be Horizontal, Vertical, or Both."
        )





