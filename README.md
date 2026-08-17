# Digital Image Processing & Matrix Operations Suite

A modular Python framework implementing foundational Digital Image Processing (DIP) concepts, matrix operations, and spatial transformations without relying on black-box high-level filters.

## Implemented Modules & Features
- **Grayscale Conversion:** Custom implementation based on the standard NTSC luminosity weighted equation ($Y = 0.299R + 0.587G + 0.114B$).
- **Spatial Sampling & Quantization:** Simulates spatial degradation through decimation sampling and discrete bit-depth reduction (e.g., 2-bit quantization).
- **Point & Intensity Transformations:** Dynamic range stretching via Logarithmic transforms, negative inversion, and Power-Law ($\gamma = 0.5, 2.0$) adjustments.
- **Affine Geometric Transformations:** 2D spatial coordinate manipulation including translation, center rotation, and scaling.
- **Neighborhood Spatial Filtering:** Spatial 2D convolution applying low-pass mean averaging and high-pass Laplacian edge kernels.
- **Distance Metrics:** Evaluation of Euclidean ($D_e$), Manhattan ($D_4$), and Chebyshev ($D_8$) spatial metrics.

## Project Structure
```text
├── dip_utils.py               # Core DIP helper functions and transformations
├── main.py                    # Main pipeline runner & visualization generator
├── requirements.txt           # Project dependencies
├── images.jpeg                # Input sample image
└── images_processed_results.png # Output visual grid
