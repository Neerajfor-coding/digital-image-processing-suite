"""
Digital Image Processing Utilities & Core Operations
Author: Neeraj Gupta
Coursework / Research Implementation
"""

import os
import cv2
import numpy as np


def load_input_image(filepath="images.jpeg", convert_to_gray=False):
    """Loads input image from disk and converts BGR to RGB/Grayscale."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Image '{filepath}' not found. Please place '{filepath}' in the project directory."
        )

    if convert_to_gray:
        return cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)

    image_bgr = cv2.imread(filepath)
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def convert_rgb_to_gray_custom(rgb_image):
    """
    Manual Grayscale conversion using standard luminosity weights:
    Gray = 0.299*R + 0.587*G + 0.114*B
    """
    if len(rgb_image.shape) == 2:
        return rgb_image

    r = rgb_image[:, :, 0]
    g = rgb_image[:, :, 1]
    b = rgb_image[:, :, 2]

    gray = 0.299 * r + 0.587 * g + 0.114 * b
    return gray.astype(np.uint8)


def apply_spatial_downsampling(gray_img, step_factor=4):
    """Reduces spatial resolution by sampling every k-th pixel."""
    return gray_img[::step_factor, ::step_factor]


def apply_intensity_quantization(gray_img, target_bits=2):
    """Simulates lower bit-depth quantization (e.g., 2-bit = 4 gray levels)."""
    total_levels = 2**target_bits
    step_size = 256 / total_levels
    quantized_img = np.floor(gray_img / step_size) * step_size
    return quantized_img.astype(np.uint8)


def compute_intensity_transforms(gray_img):
    """Applies Image Negative, Logarithmic, and Power-Law (Gamma) transformations."""
    # 1. Negative: s = 255 - r
    img_negative = 255 - gray_img

    # 2. Log Transform: s = c * log(1 + r)
    scale_constant = 255.0 / (np.log(1.0 + np.max(gray_img)) + 1e-6)
    img_log = scale_constant * (np.log(gray_img.astype(np.float64) + 1.0))
    img_log = np.array(img_log, dtype=np.uint8)

    # 3. Gamma Correction (Gamma < 1 for brightening, Gamma > 1 for contrast enhancement)
    img_gamma_low = np.array(
        255.0 * ((gray_img / 255.0) ** 0.5), dtype=np.uint8
    )
    img_gamma_high = np.array(
        255.0 * ((gray_img / 255.0) ** 2.0), dtype=np.uint8
    )

    return img_negative, img_log, img_gamma_low, img_gamma_high


def apply_geometric_transformations(rgb_img):
    """Applies Affine transformations: Translation, Rotation, and Rescaling."""
    rows, cols = rgb_img.shape[:2]

    # Translation by (+40px X, +20px Y)
    trans_matrix = np.float32([[1, 0, 40], [0, 1, 20]])
    img_translated = cv2.warpAffine(rgb_img, trans_matrix, (cols, rows))

    # Center Rotation by 45 degrees
    center = (cols // 2, rows // 2)
    rot_matrix = cv2.getRotationMatrix2D(center, 45, 1.0)
    img_rotated = cv2.warpAffine(rgb_img, rot_matrix, (cols, rows))

    # Scaling down to 70%
    img_scaled = cv2.resize(
        rgb_img,
        (int(cols * 0.7), int(rows * 0.7)),
        interpolation=cv2.INTER_LINEAR,
    )

    return img_translated, img_rotated, img_scaled


def apply_spatial_filtering(gray_img):
    """Applies Low-pass Mean Blur and High-pass Laplacian edge detection."""
    # 5x5 Box Mean Filter (Smoothing)
    mean_kernel = np.ones((5, 5), dtype=np.float32) / 25.0
    img_blurred = cv2.filter2D(gray_img, -1, mean_kernel)

    # Laplacian High-Pass Filter (Edge Extraction)
    laplacian_kernel = np.array(
        [[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32
    )
    img_edges = cv2.filter2D(gray_img, -1, laplacian_kernel)

    return img_blurred, img_edges


def calculate_pixel_distances(point_a, point_b):
    """Calculates Euclidean (De), Manhattan (D4), and Chebyshev (D8) metrics."""
    pt1 = np.array(point_a, dtype=np.float64)
    pt2 = np.array(point_b, dtype=np.float64)

    euclidean_dist = np.linalg.norm(pt1 - pt2)
    city_block_dist = np.sum(np.abs(pt1 - pt2))
    chessboard_dist = np.max(np.abs(pt1 - pt2))

    return euclidean_dist, int(city_block_dist), int(chessboard_dist)