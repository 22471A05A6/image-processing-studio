from io import BytesIO
import streamlit as st
import hashlib
import cv2
from PIL import Image, UnidentifiedImageError
import numpy as np
from processing.basic import convert_to_grayscale
from processing.transform import (
    resize_image,
    crop_image,
    rotate_image,
    flip_image
)
from processing.enhancement import (
    adjust_brightness,
    adjust_contrast,
    invert_colors
)

from processing.filters import (
    apply_blur,
    apply_gaussian_blur,
    apply_median_blur
)
from processing.edges import (
    apply_canny,
    apply_sobel
)
from processing.threshold import (
    apply_threshold,
    apply_adaptive_threshold
)


# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(
    page_title="Image Processing Studio",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------
# Main Header
# -------------------------------

# Page configuration
st.set_page_config(
    page_title="Image Processing Studio",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional page header
st.markdown(
    """
    <div style="text-align: center; padding: 15px 0 20px 0;">
        <h1 style="margin-bottom: 5px;">
            🖼️ Image Processing Studio
        </h1>
        <p style="font-size: 16px; color: #808495;">
            Upload, transform, enhance, and analyze your images
            with 15 image processing techniques.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

st.divider()


# -------------------------------
# Sidebar: Processing Methods
# -------------------------------
st.sidebar.title("⚙️ Processing Options")
st.sidebar.caption("Choose a preprocessing method")

categories = {
    "Basic": ["Grayscale", "Resize", "Crop"],
    "Transform": ["Rotate", "Flip"],
    "Enhancement": ["Brightness", "Contrast"],
    "Filters": ["Blur", "Gaussian Blur", "Median Blur"],
    "Edge Detection": ["Canny", "Sobel"],
    "Threshold": ["Threshold", "Adaptive Threshold"],
    "Color": ["Invert"]
}

selected_category = st.sidebar.selectbox(
    "Category",
    list(categories.keys())
)

selected_method = st.sidebar.radio(
    "Select Method",
    categories[selected_category]
)

st.sidebar.divider()


# -------------------------------
# Image Upload
# -------------------------------

st.sidebar.title("🎛️ Control Panel")
st.sidebar.caption("Upload and process your image")
st.sidebar.divider()
st.sidebar.subheader("📤 Image Input")

uploaded_file = st.sidebar.file_uploader(
    "📤 Upload an Image",
    type=["jpg", "jpeg", "png", "webp"],
    help="Upload a JPG, JPEG, PNG, or WEBP image."
)

# -------------------------------
# Session State Initialization
# -------------------------------

if "last_uploaded_file_id" not in st.session_state:
    st.session_state["last_uploaded_file_id"] = None

# Keys used to store processed results
result_keys = [
    "resized_image",
    "resize_dimensions",
    "cropped_image",
    "crop_dimensions",
    "rotated_image",
    "rotation_angle",
    "flipped_image",
    "flip_direction",
    "bright_image",
    "brightness_value",
    "contrast_image",
    "contrast_value",
    "blur_image",
    "blur_kernel",
    "gaussian_image",
    "gaussian_kernel",
    "median_image",
    "median_kernel",
    "canny_image",
    "canny_thresholds",
    "sobel_image",
    "threshold_image",
    "threshold_settings",
    "adaptive_image",
    "adaptive_settings",
    "inverted_image",
]

# -------------------------------
# Clear Old Results
# -------------------------------

if uploaded_file is None:
    for key in result_keys:
        st.session_state.pop(key, None)

    st.session_state["last_uploaded_file_id"] = None

# -------------------------------
# Handle Uploaded Image
# -------------------------------

if uploaded_file is not None:
    file_bytes = uploaded_file.getvalue()
    current_file_id = hashlib.md5(file_bytes).hexdigest()

    # Clear previous results if a new image is uploaded
    if current_file_id != st.session_state["last_uploaded_file_id"]:
        for key in result_keys:
            st.session_state.pop(key, None)

        st.session_state["last_uploaded_file_id"] = current_file_id

    # Read and normalize the uploaded image
    try:
        image = Image.open(BytesIO(file_bytes))
        original_image = image.convert("RGB")

        st.success("Image uploaded successfully!")

        # Show original image
        st.subheader("📷 Original Image")
        st.image(
            original_image,
            caption="Uploaded Image",
            use_container_width=True
        )

    except (ValueError, TypeError, OSError) as error:
        st.error(f"Unable to open this image: {error}")
        st.stop()

    # -------------------------------
    # Processing Options
    # -------------------------------
    # IMPORTANT:
    # Keep your existing selected_method/category selection
    # and all 15 preprocessing branches here.
    #
    # Define selected_method before using it.
    # Keep your existing image comparison code in this area.

    # -------------------------------
    # Common Download System
    # -------------------------------

    # Place this block AFTER your processing branches,
    # where selected_method and the processed result are available.

    result_key_map = {
        "Resize": "resized_image",
        "Crop": "cropped_image",
        "Rotate": "rotated_image",
        "Flip": "flipped_image",
        "Brightness": "bright_image",
        "Contrast": "contrast_image",
        "Blur": "blur_image",
        "Gaussian Blur": "gaussian_image",
        "Median Blur": "median_image",
        "Canny": "canny_image",
        "Sobel": "sobel_image",
        "Threshold": "threshold_image",
        "Adaptive Threshold": "adaptive_image",
        "Invert": "inverted_image",
    }

    download_image = None

    if selected_method == "Grayscale":
        rgb_array = np.array(original_image)
        download_image = convert_to_grayscale(rgb_array)

    elif selected_method in result_key_map:
        result_key = result_key_map[selected_method]
        download_image = st.session_state.get(result_key)

    # Display a download button for the selected result
    if download_image is not None:
        try:
            if isinstance(download_image, Image.Image):
                image_to_save = download_image
            else:
                image_array = np.asarray(download_image)

                # Convert boolean or other numeric arrays to uint8
                if image_array.dtype == np.bool_:
                    image_array = image_array.astype(np.uint8) * 255
                elif image_array.dtype != np.uint8:
                    image_array = np.clip(
                        image_array, 0, 255
                    ).astype(np.uint8)

                image_to_save = Image.fromarray(image_array)

            image_buffer = BytesIO()
            image_to_save.save(image_buffer, format="PNG")
            image_buffer.seek(0)

            st.download_button(
                label="📥 Download Processed Image",
                data=image_buffer.getvalue(),
                file_name=(
                    f"{selected_method.lower().replace(' ', '_')}.png"
                ),
                mime="image/png",
                use_container_width=True,
                key="download_processed_image"
            )

        except (ValueError, TypeError, OSError) as error:
            st.error(f"Unable to prepare the download: {error}")



# -------------------------------
# Main Workspace
# -------------------------------
st.subheader("Image Processing Workspace")

st.write(f"**Selected Category:** {selected_category}")
st.write(f"**Selected Method:** {selected_method}")

if uploaded_file is None:
    st.info("👈 Upload an image from the sidebar to begin.")
    st.caption("Supported formats: JPG, JPEG, PNG, WEBP")

else:
    try:
        image = Image.open(uploaded_file)
        image.load()

        # Normalize the image for consistent display.
        original_image = image.convert("RGB")

        st.success("Image uploaded successfully!")

        st.subheader("Original Image")

        col1, col2 = st.columns([2, 1])

        with col1:
            st.image(
                original_image,
                caption="Uploaded Image",
                use_container_width=True
            )

        with col2:
            st.markdown("### Image Information")
            st.write(f"**Filename:** {uploaded_file.name}")
            st.write(f"**Format:** {image.format or 'Unknown'}")
            st.write(f"**Width:** {image.width} px")
            st.write(f"**Height:** {image.height} px")
            st.write(f"**Color mode:** {image.mode}")

        
        st.subheader("Image Comparison")


        # Original image and processed image
        original_col, processed_col = st.columns(2)

        with original_col:
            st.markdown("### Original Image")
            st.image(
                original_image,
                use_container_width=True
            )

        with processed_col:
            st.markdown("### Processed Image")

            
            if selected_method == "Grayscale":
                rgb_array = np.array(original_image)

                processed_image = convert_to_grayscale(
                    rgb_array
                )

                st.image(
                    processed_image,
                    caption="Grayscale Image",
                    clamp=True,
                    use_container_width=True
                )

            elif selected_method == "Resize":
                st.markdown("#### Resize Settings")

                width = st.number_input(
                    "New Width (pixels)",
                    min_value=1,
                    max_value=10000,
                    value=int(original_image.width),
                    step=10
                )

                height = st.number_input(
                    "New Height (pixels)",
                    min_value=1,
                    max_value=10000,
                    value=int(original_image.height),
                    step=10
                )

                if st.button("Apply Resize", type="primary"):
                    rgb_array = np.array(original_image)

                    processed_image = resize_image(
                        rgb_array,
                        int(width),
                        int(height)
                    )

                    st.session_state["resized_image"] = (
                        processed_image
                    )
                    st.session_state["resize_dimensions"] = (
                        int(width), int(height)
                    )

                if "resized_image" in st.session_state:
                    saved_width, saved_height = (
                        st.session_state["resize_dimensions"]
                    )

                    # Display the saved result only when the
                    # current settings match the result.
                    if (int(width), int(height)) == (
                        saved_width, saved_height
                    ):
                        st.image(
                            st.session_state["resized_image"],
                            caption=f"Resized: {saved_width} × {saved_height}",
                            clamp=True,
                            use_container_width=True
                        )
                    else:
                        st.info("Click Apply Resize to update the result.")
            
            elif selected_method == "Crop":
                st.markdown("#### Crop Settings")

                img_width, img_height = original_image.size

                st.caption(
                    f"Original dimensions: {img_width} × {img_height} px"
                )

                x = st.number_input(
                    "X position",
                    min_value=0,
                    max_value=img_width - 1,
                    value=0,
                    step=1,
                    key="crop_x"
                )

                y = st.number_input(
                    "Y position",
                    min_value=0,
                    max_value=img_height - 1,
                    value=0,
                    step=1,
                    key="crop_y"
                )

                width = st.number_input(
                    "Crop width",
                    min_value=1,
                    max_value=img_width,
                    value=img_width,
                    step=1,
                    key="crop_width"
                )

                height = st.number_input(
                    "Crop height",
                    min_value=1,
                    max_value=img_height,
                    value=img_height,
                    step=1,
                    key="crop_height"
                )

                if st.button("Apply Crop", type="primary"):
                    try:
                        rgb_array = np.array(original_image)

                        cropped_image = crop_image(
                            rgb_array,
                            int(x),
                            int(y),
                            int(width),
                            int(height)
                        )

                        st.session_state["cropped_image"] = (
                            cropped_image
                        )
                        st.session_state["crop_dimensions"] = (
                            int(x), int(y), int(width), int(height)
                        )

                    except ValueError as error:
                        st.error(str(error))

                if "cropped_image" in st.session_state:
                    current_crop = (
                        int(x), int(y), int(width), int(height)
                    )

                    if current_crop == st.session_state.get(
                        "crop_dimensions"
                    ):
                        st.image(
                            st.session_state["cropped_image"],
                            caption="Cropped Image",
                            clamp=True,
                            use_container_width=True
                        )
                    else:
                        st.info("Click Apply Crop to update the result.")
            
            elif selected_method == "Rotate":
                st.markdown("#### Rotation Settings")

                angle = st.slider(
                    "Rotation Angle (degrees)",
                    min_value=-180,
                    max_value=180,
                    value=90,
                    step=1
                )

                if st.button("Apply Rotation", type="primary"):
                    rgb_array = np.array(original_image)

                    rotated_image = rotate_image(
                        rgb_array,
                        angle
                    )

                    st.session_state["rotated_image"] = rotated_image
                    st.session_state["rotation_angle"] = angle

                if "rotated_image" in st.session_state:
                    if angle == st.session_state.get("rotation_angle"):
                        st.image(
                            st.session_state["rotated_image"],
                            caption=f"Rotated by {angle}°",
                            clamp=True,
                            use_container_width=True
                        )
                    else:
                        st.info("Click Apply Rotation to update the result.")
            
            elif selected_method == "Flip":
                st.markdown("#### Flip Settings")

                direction = st.radio(
                    "Choose Flip Direction",
                    ["Horizontal", "Vertical", "Both"],
                    horizontal=True
                )

                if st.button("Apply Flip", type="primary"):
                    rgb_array = np.array(original_image)

                    flipped_image = flip_image(
                        rgb_array,
                        direction
                    )

                    st.session_state["flipped_image"] = flipped_image
                    st.session_state["flip_direction"] = direction

                if "flipped_image" in st.session_state:
                    if direction == st.session_state.get("flip_direction"):
                        st.image(
                            st.session_state["flipped_image"],
                            caption=f"{direction} Flip",
                            clamp=True,
                            use_container_width=True
                        )
                    else:
                        st.info("Click Apply Flip to update the result.")
            
            elif selected_method == "Brightness":
                st.markdown("#### Brightness Settings")

                brightness = st.slider(
                    "Adjust Brightness",
                    min_value=-100,
                    max_value=100,
                    value=0,
                    step=5,
                    help="Negative values darken; positive values brighten."
                )

                if st.button("Apply Brightness", type="primary"):
                    rgb_array = np.array(original_image)

                    bright_image = adjust_brightness(
                        rgb_array,
                        brightness
                    )

                    st.session_state["bright_image"] = bright_image
                    st.session_state["brightness_value"] = brightness

                if "bright_image" in st.session_state:
                    if brightness == st.session_state.get(
                        "brightness_value"
                    ):
                        st.image(
                            st.session_state["bright_image"],
                            caption=f"Brightness: {brightness:+d}",
                            clamp=True,
                            use_container_width=True
                        )
                    else:
                        st.info("Click Apply Brightness to update the result.")
            
            
            elif selected_method == "Contrast":
                st.markdown("#### Contrast Settings")

                contrast = st.slider(
                    "Adjust Contrast",
                    min_value=0.0,
                    max_value=3.0,
                    value=1.0,
                    step=0.1,
                    help="Below 1 lowers contrast; above 1 increases it."
                )

                if st.button("Apply Contrast", type="primary"):
                    rgb_array = np.array(original_image)

                    contrast_image = adjust_contrast(
                        rgb_array,
                        contrast
                    )

                    st.session_state["contrast_image"] = contrast_image
                    st.session_state["contrast_value"] = contrast

                if "contrast_image" in st.session_state:
                    if contrast == st.session_state.get("contrast_value"):
                        st.image(
                            st.session_state["contrast_image"],
                            caption=f"Contrast factor: {contrast:.1f}",
                            clamp=True,
                            use_container_width=True
                        )
                    else:
                        st.info("Click Apply Contrast to update the result.")
            
            elif selected_method == "Blur":
                st.markdown("#### Basic Blur Settings")

                kernel_size = st.slider(
                    "Blur Intensity",
                    min_value=3,
                    max_value=21,
                    value=5,
                    step=2,
                    help="Higher values generally create a stronger blur."
                )

                
                
                if st.button("Apply Blur", type="primary"):
                    try:
                        rgb_array = np.array(original_image)

                        blurred_image = apply_blur(
                            rgb_array,
                            kernel_size
                        )

                        if blurred_image is None or blurred_image.size == 0:
                            st.error("Blur processing returned an empty image.")
                        else:
                            st.session_state["blur_image"] = blurred_image
                            st.session_state["blur_kernel"] = kernel_size

                            st.success("Blur applied successfully!")

                    except (ValueError, cv2.error) as error:
                        st.error(f"Unable to apply blur: {error}")



                if "blur_image" in st.session_state:
                    if kernel_size == st.session_state.get("blur_kernel"):
                        st.image(
                            st.session_state["blur_image"],
                            caption=f"Basic Blur — Kernel: {kernel_size} × {kernel_size}",
                            clamp=True,
                            use_container_width=True
                        )
                    else:
                        st.info("Click Apply Blur to update the result.")
            
            elif selected_method == "Gaussian Blur":
                st.markdown("#### Gaussian Blur Settings")

                kernel_size = st.slider(
                    "Gaussian Blur Intensity",
                    min_value=3,
                    max_value=21,
                    value=5,
                    step=2,
                    help="Higher values generally create stronger blur."
                )

                if st.button("Apply Gaussian Blur", type="primary"):
                    try:
                        rgb_array = np.array(original_image)

                        gaussian_image = apply_gaussian_blur(
                            rgb_array,
                            kernel_size
                        )

                        if gaussian_image is None or gaussian_image.size == 0:
                            st.error(
                                "Gaussian Blur returned an empty image."
                            )
                        else:
                            st.session_state["gaussian_image"] = gaussian_image
                            st.session_state["gaussian_kernel"] = kernel_size

                            st.success(
                                "Gaussian Blur applied successfully!"
                            )

                    except (ValueError, cv2.error) as error:
                        st.error(
                            f"Unable to apply Gaussian Blur: {error}"
                        )

                if "gaussian_image" in st.session_state:
                    if kernel_size == st.session_state.get("gaussian_kernel"):
                        st.image(
                            st.session_state["gaussian_image"],
                            caption=f"Gaussian Blur — Kernel: {kernel_size} × {kernel_size}",
                            clamp=True,
                            use_container_width=True
                        )
                    else:
                        st.info("Click Apply Gaussian Blur to update the result.")
            
            elif selected_method == "Median Blur":
                st.markdown("#### Median Blur Settings")

                kernel_size = st.slider(
                    "Median Blur Intensity",
                    min_value=3,
                    max_value=21,
                    value=5,
                    step=2,
                    help="Higher values generally produce stronger smoothing."
                )

                
                if st.button("Apply Median Blur", type="primary"):
                    try:
                        rgb_array = np.array(original_image)

                        median_image = apply_median_blur(
                            rgb_array,
                            kernel_size
                        )

                        if median_image is None or median_image.size == 0:
                            st.error(
                                "Median Blur returned an empty image."
                            )
                        else:
                            st.session_state["median_image"] = median_image
                            st.session_state["median_kernel"] = kernel_size

                            st.success(
                                "Median Blur applied successfully!"
                            )

                    except (ValueError, cv2.error) as error:
                        st.error(
                            f"Unable to apply Median Blur: {error}"
                        )


                if "median_image" in st.session_state:
                    if kernel_size == st.session_state.get("median_kernel"):
                        st.image(
                            st.session_state["median_image"],
                            caption=f"Median Blur — Kernel: {kernel_size} × {kernel_size}",
                            clamp=True,
                            use_container_width=True
                        )
                    else:
                        st.info("Click Apply Median Blur to update the result.")
            
            elif selected_method == "Canny":
                st.markdown("#### Canny Edge Detection Settings")

                threshold1 = st.slider(
                    "Lower Threshold",
                    min_value=0,
                    max_value=254,
                    value=100
                )

                threshold2 = st.slider(
                    "Upper Threshold",
                    min_value=1,
                    max_value=255,
                    value=200
                )

                
                if threshold1 >= threshold2:
                    st.warning(
                        "Upper Threshold must be greater than Lower Threshold."
                    )

                elif st.button("Apply Canny", type="primary"):
                    try:
                        rgb_array = np.array(original_image)

                        canny_image = apply_canny(
                            rgb_array,
                            threshold1,
                            threshold2
                        )

                        if canny_image is None or canny_image.size == 0:
                            st.error(
                                "Canny detection returned an empty image."
                            )
                        else:
                            st.session_state["canny_image"] = canny_image
                            st.session_state["canny_thresholds"] = (
                                threshold1,
                                threshold2
                            )

                            st.success(
                                "Canny Edge Detection completed successfully!"
                            )

                    except (ValueError, cv2.error) as error:
                        st.error(
                            f"Unable to apply Canny detection: {error}"
                        )


                if "canny_image" in st.session_state:
                    if (
                        threshold1,
                        threshold2
                    ) == st.session_state.get("canny_thresholds"):
                        st.image(
                            st.session_state["canny_image"],
                            caption="Canny Edge Detection",
                            clamp=True,
                            use_container_width=True
                        )
                    else:
                        st.info("Click Apply Canny to update the result.")
            
            elif selected_method == "Sobel":
                st.markdown("#### Sobel Edge Detection")

                st.caption(
                    "Detect horizontal and vertical intensity changes."
                )

                if st.button("Apply Sobel", type="primary"):
                    rgb_array = np.array(original_image)

                    sobel_image = apply_sobel(rgb_array)

                    st.session_state["sobel_image"] = sobel_image

                if "sobel_image" in st.session_state:
                    st.image(
                        st.session_state["sobel_image"],
                        caption="Sobel Edge Detection",
                        clamp=True,
                        use_container_width=True
                    )
            
            elif selected_method == "Threshold":
                st.markdown("#### Threshold Settings")

                threshold_value = st.slider(
                    "Threshold Value",
                    min_value=0,
                    max_value=255,
                    value=127,
                    step=1
                )

                max_value = st.slider(
                    "Maximum Pixel Value",
                    min_value=1,
                    max_value=255,
                    value=255
                )

                if st.button("Apply Threshold", type="primary"):
                    rgb_array = np.array(original_image)

                    threshold_image = apply_threshold(
                        rgb_array,
                        threshold_value,
                        max_value
                    )

                    st.session_state["threshold_image"] = threshold_image
                    st.session_state["threshold_settings"] = (
                        threshold_value,
                        max_value
                    )

                if "threshold_image" in st.session_state:
                    if (
                        threshold_value,
                        max_value
                    ) == st.session_state.get("threshold_settings"):
                        st.image(
                            st.session_state["threshold_image"],
                            caption=f"Binary Threshold: {threshold_value}",
                            clamp=True,
                            use_container_width=True
                        )
                    else:
                        st.info("Click Apply Threshold to update the result.")
            
            elif selected_method == "Adaptive Threshold":
                st.markdown("#### Adaptive Threshold Settings")

                block_size = st.slider(
                    "Block Size",
                    min_value=3,
                    max_value=51,
                    value=11,
                    step=2,
                    help="Must be an odd number. Controls local neighborhood size."
                )

                c_value = st.slider(
                    "Constant C",
                    min_value=-10,
                    max_value=20,
                    value=2,
                    step=1,
                    help="Subtracted from the local threshold."
                )

                if st.button("Apply Adaptive Threshold", type="primary"):
                    rgb_array = np.array(original_image)

                    adaptive_image = apply_adaptive_threshold(
                        rgb_array,
                        255,
                        block_size,
                        c_value
                    )

                    st.session_state["adaptive_image"] = adaptive_image
                    st.session_state["adaptive_settings"] = (
                        block_size,
                        c_value
                    )

                if "adaptive_image" in st.session_state:
                    if (
                        block_size,
                        c_value
                    ) == st.session_state.get("adaptive_settings"):
                        st.image(
                            st.session_state["adaptive_image"],
                            caption=f"Adaptive Threshold — Block: {block_size}, C: {c_value}",
                            clamp=True,
                            use_container_width=True
                        )
                    else:
                        st.info(
                            "Click Apply Adaptive Threshold to update the result."
                        )
            
            elif selected_method == "Invert":
                st.markdown("#### Invert Colors")

                st.caption(
                    "Convert every pixel to its complementary intensity."
                )

                if st.button("Apply Inversion", type="primary"):
                    rgb_array = np.array(original_image)

                    inverted_image = invert_colors(rgb_array)

                    st.session_state["inverted_image"] = inverted_image

                if "inverted_image" in st.session_state:
                    st.image(
                        st.session_state["inverted_image"],
                        caption="Inverted Image",
                        clamp=True,
                        use_container_width=True
                    )


            else:
                st.info(
                    f"{selected_method} processing "
                    "will be added in a future step."
                )


        st.divider()

        st.subheader("Image Information")

        info1, info2, info3 = st.columns(3)

        with info1:
            st.metric("Width", f"{image.width} px")

        with info2:
            st.metric("Height", f"{image.height} px")

        with info3:
            st.metric("Format", image.format or "Unknown")


    except (UnidentifiedImageError, OSError, ValueError):
        st.error(
            "Unable to read this image. Please upload a valid image file."
        )

# -------------------------------
# Footer
# -------------------------------
st.divider()
st.caption(
    "Image Processing Studio | Built with Streamlit, Pillow and OpenCV"
)
