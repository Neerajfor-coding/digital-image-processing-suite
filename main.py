"""
Main Execution Script for Digital Image Processing Suite
Author: Neeraj Gupta
"""

import dip_utils as dip
import matplotlib.pyplot as plt


def run_pipeline():
    input_image_path = "images.jpeg"
    print("=" * 60)
    print(f"Starting DIP Processing Pipeline for '{input_image_path}'...")
    print("=" * 60)

    # 1. Load image and perform custom grayscale conversion
    img_rgb = dip.load_input_image(input_image_path)
    img_gray = dip.convert_rgb_to_gray_custom(img_rgb)
    print(
        f"[✓] Image loaded successfully with dimensions: {img_rgb.shape[:2]}"
    )

    # 2. Downsampling and Bit-Depth Quantization
    downsampled_4x = dip.apply_spatial_downsampling(img_gray, step_factor=4)
    quantized_2bit = dip.apply_intensity_quantization(img_gray, target_bits=2)
    print("[✓] Sampling and Quantization operations completed.")

    # 3. Intensity Transformations
    img_neg, img_log, gamma_05, gamma_20 = dip.compute_intensity_transforms(
        img_gray
    )
    print("[✓] Intensity transformations computed.")

    # 4. Geometric Transformations
    trans, rot, scaled = dip.apply_geometric_transformations(img_rgb)
    print("[✓] Geometric operations executed.")

    # 5. Spatial Filtering
    blurred, edges = dip.apply_spatial_filtering(img_gray)
    print("[✓] Spatial filtering and edge detection completed.")

    # 6. Distance Metrics between two sample coordinates
    h, w = img_gray.shape
    p1 = (h // 4, w // 4)
    p2 = (3 * h // 4, 3 * w // 4)
    d_euc, d_man, d_cheb = dip.calculate_pixel_distances(p1, p2)

    print("\n--- Pixel Neighborhood Distance Evaluation ---")
    print(f"Coordinate P1: {p1} | Coordinate P2: {p2}")
    print(f"  • Euclidean Distance (De)  : {d_euc:.2f} pixels")
    print(f"  • City-Block Distance (D4) : {d_man} pixels")
    print(f"  • Chessboard Distance (D8) : {d_cheb} pixels")
    print("-" * 46)

    # 7. Visualization Grid Generation
    plt.figure(figsize=(15, 10))

    displays = [
        ("Original RGB", img_rgb, None),
        ("Grayscale (Luminosity)", img_gray, "gray"),
        ("4x Downsampled", downsampled_4x, "gray"),
        ("2-Bit Quantization", quantized_2bit, "gray"),
        ("Image Negative", img_neg, "gray"),
        ("Log Transformation", img_log, "gray"),
        ("Gamma = 0.5 (Brightened)", gamma_05, "gray"),
        ("Gamma = 2.0 (High Contrast)", gamma_20, "gray"),
        ("Rotated 45°", rot, None),
        ("Translated (+40, +20)", trans, None),
        ("5x5 Mean Box Blur", blurred, "gray"),
        ("Laplacian Edge Filter", edges, "gray"),
    ]

    for idx, (title, img, colormap) in enumerate(displays, start=1):
        plt.subplot(3, 4, idx)
        plt.title(title, fontsize=10, fontweight="bold")
        plt.imshow(img, cmap=colormap)
        plt.axis("off")

    plt.tight_layout()
    output_filename = "images_processed_results.png"
    plt.savefig(output_filename, dpi=300)
    print(f"\n[✓] Results visualization saved as '{output_filename}'.")
    plt.show()


if __name__ == "__main__":
    run_pipeline()
