
# 🖼️ Image Processing Studio

A Python-based image processing application built with Streamlit, OpenCV, NumPy, and Pillow. It provides an interactive interface to upload images, apply image processing techniques, compare results, and download processed images.

## ✨ Features

The application supports 15 image processing techniques:

### Basic Operations
- Grayscale Conversion
- Image Resizing
- Image Cropping

### Transformations
- Image Rotation
- Image Flipping

### Image Enhancement
- Brightness Adjustment
- Contrast Adjustment

### Filters
- Blur
- Gaussian Blur
- Median Blur

### Edge Detection
- Canny Edge Detection
- Sobel Edge Detection

### Thresholding
- Threshold
- Adaptive Threshold

### Color Operations
- Image Inversion

## 🛠️ Technologies Used

- Python
- Streamlit
- OpenCV
- NumPy
- Pillow

## 📁 Project Structure

```text
image_processing_app/
├── app.py
├── processing/
│   ├── __init__.py
│   ├── basic.py
│   ├── transform.py
│   ├── enhancement.py
│   ├── filters.py
│   ├── edges.py
│   └── threshold.py
├── utils/
│   ├── __init__.py
│   └── image_utils.py
├── assets/
├── requirements.txt
└── README.md
```

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd image_processing_app
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

### 3. Activate the virtual environment

Windows Command Prompt:

```bash
venv\Scripts\activate
```

PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the application

```bash
streamlit run app.py
```

The application will open in your browser.

## 🚀 How to Use

1. Launch the application.
2. Upload an image in JPG, JPEG, PNG, or WEBP format.
3. Select an image processing technique.
4. Adjust the available settings.
5. Apply the selected operation.
6. Compare the original and processed images.
7. Download the processed image as a PNG file.

## 🎯 Project Objective

The objective of this project is to demonstrate fundamental image processing techniques through an interactive web application built using Python and computer vision libraries.

## 👨‍💻 Author

Add your name and GitHub profile here.

## 📄 License

This project is available for educational and learning purposes.
