import argparse
import math
import os
import sys
from typing import Dict, Tuple
import numpy as np

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


# =============================================================================
# 1. MATHEMATICAL KERNEL DEFINITIONS & DERIVATIONS
# =============================================================================

def get_sobel_kernels() -> Tuple[np.ndarray, np.ndarray]:
    gx = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ], dtype=np.float64)
    gy = np.array([
        [-1, -2, -1],
        [ 0,  0,  0],
        [ 1,  2,  1]
    ], dtype=np.float64)
    return gx, gy


def get_prewitt_kernels() -> Tuple[np.ndarray, np.ndarray]:
    gx = np.array([
        [-1, 0, 1],
        [-1, 0, 1],
        [-1, 0, 1]
    ], dtype=np.float64)
    gy = np.array([
        [-1, -1, -1],
        [ 0,  0,  0],
        [ 1,  1,  1]
    ], dtype=np.float64)
    return gx, gy


def get_kirsch_kernels() -> Dict[str, np.ndarray]:
    return {
        "k0_North (0°)": np.array([
            [ 5,  5,  5],
            [-3,  0, -3],
            [-3, -3, -3]
        ], dtype=np.float64),
        "k1_NorthWest (45°)": np.array([
            [ 5,  5, -3],
            [ 5,  0, -3],
            [-3, -3, -3]
        ], dtype=np.float64),
        "k2_West (90°)": np.array([
            [ 5, -3, -3],
            [ 5,  0, -3],
            [ 5, -3, -3]
        ], dtype=np.float64),
        "k3_SouthWest (135°)": np.array([
            [-3, -3, -3],
            [ 5,  0, -3],
            [ 5,  5, -3]
        ], dtype=np.float64),
        "k4_South (180°)": np.array([
            [-3, -3, -3],
            [-3,  0, -3],
            [ 5,  5,  5]
        ], dtype=np.float64),
        "k5_SouthEast (225°)": np.array([
            [-3, -3, -3],
            [-3,  0,  5],
            [-3,  5,  5]
        ], dtype=np.float64),
        "k6_East (270°)": np.array([
            [-3, -3,  5],
            [-3,  0,  5],
            [-3, -3,  5]
        ], dtype=np.float64),
        "k7_NorthEast (315°)": np.array([
            [-3,  5,  5],
            [-3,  0,  5],
            [-3, -3, -3]
        ], dtype=np.float64),
    }


def generate_gaussian_kernel(size: int, sigma: float) -> np.ndarray:
    """
    Computes a normalized 2D Gaussian smoothing kernel G_sigma(x, y):
        G_sigma(x, y) = (1 / (2 * pi * sigma^2)) * exp(-(x^2 + y^2) / (2 * sigma^2))
    """
    if size % 2 == 0:
        size += 1
    half = size // 2
    y, x = np.mgrid[-half:half + 1, -half:half + 1]
    r2 = x**2 + y**2
    coeff = 1.0 / (2.0 * math.pi * (sigma ** 2))
    kernel = coeff * np.exp(-r2 / (2.0 * (sigma ** 2)))
    return kernel / np.sum(kernel)


def generate_log_kernel(size: int, sigma: float, zero_mean: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes the 2D Laplacian of Gaussian (LoG) kernel:
      LoG(x, y) = -(1 / (pi*sigma^4)) * [1 - (x^2 + y^2)/(2*sigma^2)] * exp(-(x^2 + y^2)/(2*sigma^2))
    """
    if size % 2 == 0:
        size += 1
    half = size // 2
    y, x = np.mgrid[-half:half + 1, -half:half + 1]
    r2 = x**2 + y**2
    sigma2 = sigma ** 2
    sigma4 = sigma ** 4

    raw_kernel = -(1.0 / (math.pi * sigma4)) * (1.0 - r2 / (2.0 * sigma2)) * np.exp(-r2 / (2.0 * sigma2))
    zm_kernel = (raw_kernel - np.mean(raw_kernel)) if zero_mean else raw_kernel.copy()
    return raw_kernel, zm_kernel


def get_discrete_mexican_hat_5x5() -> np.ndarray:
    return np.array([
        [ 0,  0, -1,  0,  0],
        [ 0, -1, -2, -1,  0],
        [-1, -2, 16, -2, -1],
        [ 0, -1, -2, -1,  0],
        [ 0,  0, -1,  0,  0]
    ], dtype=np.float64)


def generate_dog_kernel(size: int, sigma1: float, sigma2: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes the 2D Difference of Gaussians (DoG) kernel:
      DoG(x, y; sigma1, sigma2) = G_sigma1(x, y) - G_sigma2(x, y)
    """
    g1 = generate_gaussian_kernel(size, sigma1)
    g2 = generate_gaussian_kernel(size, sigma2)
    dog = g1 - g2
    return g1, g2, dog


# =============================================================================
# 2. 2D SPATIAL CONVOLUTION & ZERO-CROSSING DETECTION
# =============================================================================

def convolve2d_replicate(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    img = image.astype(np.float64)
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2
    padded = np.pad(img, ((pad_h, pad_h), (pad_w, pad_w)), mode="edge")

    shape = (img.shape[0], img.shape[1], kh, kw)
    strides = (padded.strides[0], padded.strides[1], padded.strides[0], padded.strides[1])
    windows = np.lib.stride_tricks.as_strided(padded, shape=shape, strides=strides)
    return np.einsum("ijkl,kl->ij", windows, kernel)


def detect_zero_crossings(response: np.ndarray, slope_threshold: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    h, w = response.shape
    padded = np.pad(response, ((1, 1), (1, 1)), mode="edge")

    pairs = [
        (padded[1:h+1, 0:w],   padded[1:h+1, 2:w+2]), # Left-Right
        (padded[0:h,   1:w+1], padded[2:h+2, 1:w+1]), # Up-Down
        (padded[0:h,   0:w],   padded[2:h+2, 2:w+2]), # Main Diagonal
        (padded[0:h,   2:w+2], padded[2:h+2, 0:w]),   # Anti-Diagonal
    ]

    zc_strength = np.zeros((h, w), dtype=np.float64)
    for p, q in pairs:
        opposite_sign = (p * q) < 0.0
        diff = np.abs(p - q)
        zc_strength = np.maximum(zc_strength, np.where(opposite_sign, diff, 0.0))

    binary_edges = np.where(zc_strength > slope_threshold, 255, 0).astype(np.uint8)
    return zc_strength, binary_edges


# =============================================================================
# 3. COMPLETE EDGE SEGMENTATION PIPELINES
# =============================================================================

def run_sobel_segmentation(image: np.ndarray, threshold: float = 80.0) -> Dict[str, np.ndarray]:
    gx_kernel, gy_kernel = get_sobel_kernels()
    gx = convolve2d_replicate(image, gx_kernel)
    gy = convolve2d_replicate(image, gy_kernel)
    magnitude = np.hypot(gx, gy)
    direction_deg = np.degrees(np.arctan2(gy, gx))
    binary_edges = np.where(magnitude >= threshold, 255, 0).astype(np.uint8)
    return {
        "Gx": gx,
        "Gy": gy,
        "magnitude": magnitude,
        "direction_deg": direction_deg,
        "binary_edges": binary_edges,
    }


def run_prewitt_segmentation(image: np.ndarray, threshold: float = 60.0) -> Dict[str, np.ndarray]:
    gx_kernel, gy_kernel = get_prewitt_kernels()
    gx = convolve2d_replicate(image, gx_kernel)
    gy = convolve2d_replicate(image, gy_kernel)
    magnitude = np.hypot(gx, gy)
    direction_deg = np.degrees(np.arctan2(gy, gx))
    binary_edges = np.where(magnitude >= threshold, 255, 0).astype(np.uint8)
    return {
        "Gx": gx,
        "Gy": gy,
        "magnitude": magnitude,
        "direction_deg": direction_deg,
        "binary_edges": binary_edges,
    }


def run_kirsch_segmentation(image: np.ndarray, threshold: float = 180.0) -> Dict[str, np.ndarray]:
    kernels = get_kirsch_kernels()
    responses = [convolve2d_replicate(image, kernels[k_name]) for k_name in kernels]
    stack = np.stack(responses, axis=0)
    abs_stack = np.abs(stack)
    max_magnitude = np.max(abs_stack, axis=0)
    best_dir_idx = np.argmax(stack, axis=0)
    direction_deg = best_dir_idx * 45.0
    binary_edges = np.where(max_magnitude >= threshold, 255, 0).astype(np.uint8)
    return {
        "responses_stack": stack,
        "magnitude": max_magnitude,
        "direction_deg": direction_deg,
        "binary_edges": binary_edges,
    }


def run_log_segmentation(
    image: np.ndarray,
    kernel_size: int = 5,
    sigma: float = 1.0,
    zc_threshold: float = 1.5,
    use_discrete_mexican_hat: bool = False
) -> Dict[str, np.ndarray]:
    if use_discrete_mexican_hat:
        raw_kernel = get_discrete_mexican_hat_5x5()
        kernel = raw_kernel
    else:
        raw_kernel, kernel = generate_log_kernel(kernel_size, sigma, zero_mean=True)

    response = convolve2d_replicate(image, kernel)
    zc_strength, binary_edges = detect_zero_crossings(response, slope_threshold=zc_threshold)
    return {
        "raw_kernel": raw_kernel,
        "kernel": kernel,
        "response": response,
        "zc_strength": zc_strength,
        "binary_edges": binary_edges,
    }


def run_dog_segmentation(
    image: np.ndarray,
    kernel_size: int = 5,
    sigma1: float = 1.0,
    sigma2: float = 1.6,
    zc_threshold: float = 1.5
) -> Dict[str, np.ndarray]:
    g1, g2, dog_kernel = generate_dog_kernel(kernel_size, sigma1, sigma2)
    smooth1 = convolve2d_replicate(image, g1)
    smooth2 = convolve2d_replicate(image, g2)
    response = smooth1 - smooth2
    zc_strength, binary_edges = detect_zero_crossings(response, slope_threshold=zc_threshold)
    return {
        "G_sigma1": g1,
        "G_sigma2": g2,
        "kernel": dog_kernel,
        "smooth1": smooth1,
        "smooth2": smooth2,
        "response": response,
        "zc_strength": zc_strength,
        "binary_edges": binary_edges,
    }


# =============================================================================
# 4. STEP-BY-STEP MATHEMATICAL WORKED CALCULATIONS ON A 5x5 IMAGE PATCH
# =============================================================================

def print_step_by_step_mathematics(
    patch_5x5: np.ndarray,
    sigma_log: float = 1.0,
    sigma1_dog: float = 1.0,
    sigma2_dog: float = 1.6,
    patch_label: str = "Center (2, 2)"
) -> None:
    assert patch_5x5.shape == (5, 5), "Step-by-step mathematical walkthrough expects a 5x5 matrix."
    inner_3x3 = patch_5x5[1:4, 1:4]

    sep = "=" * 78
    sub = "-" * 78
    print(sep)
    print(f"DIGITAL IMAGE PROCESSING — STEP-BY-STEP MATH CALCULATIONS [{patch_label}]")
    print(sep)
    print("\n[INPUT] 5x5 Grayscale Image Patch I(x, y) (center pixel at index (2, 2)):")
    print(np.array2string(patch_5x5, formatter={"float_kind": lambda v: f"{v:6.1f}"}))
    print("\n[INNER 3x3 WINDOW] Centered at (2, 2) for 3x3 First-Order & Compass Masks:")
    print(np.array2string(inner_3x3, formatter={"float_kind": lambda v: f"{v:6.1f}"}))

    # 1. SOBEL
    print(f"\n{sub}")
    print("1. SOBEL OPERATOR (First-Order Gradient)")
    print(sub)
    sobel_gx, sobel_gy = get_sobel_kernels()
    s_gx_val = float(np.sum(inner_3x3 * sobel_gx))
    s_gy_val = float(np.sum(inner_3x3 * sobel_gy))
    s_mag = math.hypot(s_gx_val, s_gy_val)
    s_l1 = abs(s_gx_val) + abs(s_gy_val)
    s_theta = math.degrees(math.atan2(s_gy_val, s_gx_val))
    print(f"  Gx = ({inner_3x3[0,2]:.0f} + 2*{inner_3x3[1,2]:.0f} + {inner_3x3[2,2]:.0f}) "
          f"- ({inner_3x3[0,0]:.0f} + 2*{inner_3x3[1,0]:.0f} + {inner_3x3[2,0]:.0f}) = {s_gx_val:.2f}")
    print(f"  Gy = ({inner_3x3[2,0]:.0f} + 2*{inner_3x3[2,1]:.0f} + {inner_3x3[2,2]:.0f}) "
          f"- ({inner_3x3[0,0]:.0f} + 2*{inner_3x3[0,1]:.0f} + {inner_3x3[0,2]:.0f}) = {s_gy_val:.2f}")
    print(f"  Gradient Magnitude |G| = sqrt(Gx^2 + Gy^2) = {s_mag:.4f} (L1 = {s_l1:.2f}, theta = {s_theta:.2f} deg)")

    # 2. PREWITT
    print(f"\n{sub}")
    print("2. PREWITT OPERATOR (First-Order Gradient)")
    print(sub)
    prew_gx, prew_gy = get_prewitt_kernels()
    p_gx_val = float(np.sum(inner_3x3 * prew_gx))
    p_gy_val = float(np.sum(inner_3x3 * prew_gy))
    p_mag = math.hypot(p_gx_val, p_gy_val)
    p_theta = math.degrees(math.atan2(p_gy_val, p_gx_val))
    print(f"  Gx = ({inner_3x3[0,2]:.0f} + {inner_3x3[1,2]:.0f} + {inner_3x3[2,2]:.0f}) "
          f"- ({inner_3x3[0,0]:.0f} + {inner_3x3[1,0]:.0f} + {inner_3x3[2,0]:.0f}) = {p_gx_val:.2f}")
    print(f"  Gy = ({inner_3x3[2,0]:.0f} + {inner_3x3[2,1]:.0f} + {inner_3x3[2,2]:.0f}) "
          f"- ({inner_3x3[0,0]:.0f} + {inner_3x3[0,1]:.0f} + {inner_3x3[0,2]:.0f}) = {p_gy_val:.2f}")
    print(f"  Gradient Magnitude |G| = sqrt(Gx^2 + Gy^2) = {p_mag:.4f} (theta = {p_theta:.2f} deg)")

    # 3. KIRSCH COMPASS OPERATOR
    print(f"\n{sub}")
    print("3. KIRSCH COMPASS OPERATOR (8 Directional Masks k0..k7)")
    print(sub)
    kirsch_kernels = get_kirsch_kernels()
    k_vals = {}
    for k_name, k_mat in kirsch_kernels.items():
        val = float(np.sum(inner_3x3 * k_mat))
        k_vals[k_name] = val
        print(f"  Response {k_name:22s}: sum(I_3x3 * {k_name[:2]}) = {val:8.2f}")
    max_k_name = max(k_vals, key=lambda k: abs(k_vals[k]))
    print(f"  Kirsch Maximum Magnitude M(2,2) = max_k |R_k| = {abs(k_vals[max_k_name]):.2f} "
          f"(Dominant Direction: {max_k_name})")

    # 4. LAPLACIAN OF GAUSSIAN (LoG) — MAIN FOCUS
    print(f"\n{sub}")
    print("4. LAPLACIAN OF GAUSSIAN (LoG — Marr-Hildreth Second-Order Operator) [MAIN FOCUS]")
    print(sub)
    mex_hat = get_discrete_mexican_hat_5x5()
    mex_resp_center = float(np.sum(patch_5x5 * mex_hat))
    raw_log, zm_log = generate_log_kernel(5, sigma_log, zero_mean=True)
    log_resp_center = float(np.sum(patch_5x5 * zm_log))

    print("  (4a) Classic 5x5 Discrete Mexican Hat Kernel (sum = 0):")
    print(np.array2string(mex_hat, formatter={"float_kind": lambda v: f"{v:5.0f}"}))
    print(f"       Discrete Mexican Hat Response at (2,2) = sum(I_5x5 * K_MexicanHat) = {mex_resp_center:.4f}")

    print(f"\n  (4b) Analytical Zero-Mean 5x5 LoG Kernel (sigma = {sigma_log:.2f}, sum = {np.sum(zm_log):.2e}):")
    print(np.array2string(zm_log, formatter={"float_kind": lambda v: f"{v:8.4f}"}))
    print(f"       Analytical LoG Response at (2,2) = sum(I_5x5 * K_LoG) = {log_resp_center:.6f}")

    full_log_res = run_log_segmentation(patch_5x5, kernel_size=5, sigma=sigma_log, zc_threshold=1.0)
    r_log_map = full_log_res["response"]
    print("\n  (4c) Full 5x5 LoG Response Map R_LoG(x, y) & Zero-Crossing Check around (2,2):")
    print(np.array2string(r_log_map, formatter={"float_kind": lambda v: f"{v:8.3f}"}))
    print(f"       Zero-Crossing Slope Strength at (2,2) = {full_log_res['zc_strength'][2,2]:.4f}")

    # 5. DIFFERENCE OF GAUSSIANS (DoG) — MAIN FOCUS
    print(f"\n{sub}")
    print("5. DIFFERENCE OF GAUSSIANS (DoG — Band-Pass LoG Approximation) [MAIN FOCUS]")
    print(sub)
    g1, g2, dog_k = generate_dog_kernel(5, sigma1_dog, sigma2_dog)
    s1_center = float(np.sum(patch_5x5 * g1))
    s2_center = float(np.sum(patch_5x5 * g2))
    dog_center = float(np.sum(patch_5x5 * dog_k))

    print(f"  (5a) 5x5 DoG Kernel = G_sigma1({sigma1_dog:.2f}) - G_sigma2({sigma2_dog:.2f}) (sum = {np.sum(dog_k):.2e}):")
    print(np.array2string(dog_k, formatter={"float_kind": lambda v: f"{v:8.4f}"}))
    print(f"  (5b) Convolution at Center Pixel (2,2):")
    print(f"       I_sigma1(2,2) = {s1_center:.6f},  I_sigma2(2,2) = {s2_center:.6f}")
    print(f"       DoG(2,2) = I_sigma1(2,2) - I_sigma2(2,2) = {dog_center:.6f}")

    full_dog_res = run_dog_segmentation(patch_5x5, kernel_size=5, sigma1=sigma1_dog, sigma2=sigma2_dog, zc_threshold=0.5)
    print(f"       Zero-Crossing Slope Strength at (2,2) = {full_dog_res['zc_strength'][2,2]:.4f}")
    print(sep)


# =============================================================================
# 5. IMAGE LOADING, SYNTHETIC FALLBACK & CLI ENTRY POINT
# =============================================================================

def create_synthetic_test_image(height: int = 256, width: int = 256) -> np.ndarray:
    img = np.full((height, width), 30.0, dtype=np.float64)
    img[35:115, 30:115] = 210.0
    img[60:90, 55:90] = 65.0

    y, x = np.ogrid[:height, :width]
    r_circle = np.sqrt((x - 185)**2 + (y - 75)**2)
    img[r_circle <= 45] = 230.0
    img[r_circle <= 22] = 95.0

    for row in range(140, 230):
        col_end = 35 + (row - 140)
        img[row, 35:min(col_end, 130)] = 195.0

    for idx, col_start in enumerate(range(145, 235, 18)):
        img[145:230, col_start:col_start + 18] = 225.0 if idx % 2 == 0 else 50.0

    return np.clip(img, 0, 255).astype(np.uint8)


def extract_5x5_patch_from_image(img: np.ndarray, cx: int, cy: int) -> np.ndarray:
    h, w = img.shape
    padded = np.pad(img.astype(np.float64), ((2, 2), (2, 2)), mode="edge")
    return padded[cy:cy + 5, cx:cx + 5]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Digital Image Processing: Edge-Based Segmentation from Image Input (Sobel, Prewitt, Kirsch, LoG, DoG)"
    )
    parser.add_argument("--image", type=str, default=None, help="Path to input image file (PNG, JPG, BMP, etc.)")
    parser.add_argument("--sigma-log", type=float, default=1.0, help="Sigma for LoG operator (default: 1.0)")
    parser.add_argument("--sigma1-dog", type=float, default=1.0, help="Sigma1 for DoG operator (default: 1.0)")
    parser.add_argument("--sigma2-dog", type=float, default=1.6, help="Sigma2 for DoG operator (default: 1.6)")
    parser.add_argument("--save-dir", type=str, default="segmented_outputs", help="Directory to save segmented output images")
    args = parser.parse_args()

    image_path = args.image
    if image_path is None and sys.stdin.isatty():
        user_inp = input("Enter input image file path (or press Enter to use built-in DIP test image): ").strip()
        if user_inp:
            image_path = user_inp

    if image_path and os.path.exists(image_path) and HAS_CV2:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Warning: Could not decode image at '{image_path}'. Using synthetic DIP test image.")
            img = create_synthetic_test_image()
            source_desc = "Built-in Synthetic DIP Test Image (256x256)"
        else:
            source_desc = f"Loaded Input Image '{image_path}' ({img.shape[1]}x{img.shape[0]})"
    else:
        img = create_synthetic_test_image()
        source_desc = "Built-in Synthetic DIP Test Image (256x256)"

    print(f"\n[IMAGE INPUT SOURCE]: {source_desc}")

    # Run all 5 edge-based segmentation operators on the input image
    sobel_out = run_sobel_segmentation(img, threshold=80.0)
    prewitt_out = run_prewitt_segmentation(img, threshold=60.0)
    kirsch_out = run_kirsch_segmentation(img, threshold=180.0)
    log_out = run_log_segmentation(img, kernel_size=5, sigma=args.sigma_log, zc_threshold=1.5)
    dog_out = run_dog_segmentation(img, kernel_size=5, sigma1=args.sigma1_dog, sigma2=args.sigma2_dog, zc_threshold=1.5)

    # If a custom image was provided, extract the 5x5 neighborhood around its strongest edge pixel;
    # otherwise also show the canonical 5x5 step-edge patch.
    if image_path and os.path.exists(image_path):
        peak_y, peak_x = np.unravel_index(np.argmax(sobel_out["magnitude"]), img.shape)
        patch_5x5 = extract_5x5_patch_from_image(img, int(peak_x), int(peak_y))
        patch_label = f"Extracted 5x5 Patch from '{os.path.basename(image_path)}' at Edge Pixel ({peak_x}, {peak_y})"
    else:
        patch_5x5 = np.array([
            [ 20,  20, 100, 200, 200],
            [ 20,  25, 110, 200, 200],
            [ 20,  30, 120, 205, 210],
            [ 25,  30, 115, 200, 210],
            [ 25,  35, 120, 205, 210]
        ], dtype=np.float64)
        patch_label = "Sample 5x5 Step-Edge Patch at Center (2, 2)"

    print_step_by_step_mathematics(
        patch_5x5,
        sigma_log=args.sigma_log,
        sigma1_dog=args.sigma1_dog,
        sigma2_dog=args.sigma2_dog,
        patch_label=patch_label
    )

    print("\nFULL IMAGE EDGE SEGMENTATION SUMMARY (Image Shape: {}x{}):".format(img.shape[0], img.shape[1]))
    print(f"  - Sobel   Edge Pixels Detected : {int(np.sum(sobel_out['binary_edges'] > 0)):6d} px (Max |G| = {np.max(sobel_out['magnitude']):.2f})")
    print(f"  - Prewitt Edge Pixels Detected : {int(np.sum(prewitt_out['binary_edges'] > 0)):6d} px (Max |G| = {np.max(prewitt_out['magnitude']):.2f})")
    print(f"  - Kirsch  Edge Pixels Detected : {int(np.sum(kirsch_out['binary_edges'] > 0)):6d} px (Max  M  = {np.max(kirsch_out['magnitude']):.2f})")
    print(f"  - LoG     Zero-Crossing Pixels : {int(np.sum(log_out['binary_edges'] > 0)):6d} px (Max ZC  = {np.max(log_out['zc_strength']):.4f})")
    print(f"  - DoG     Zero-Crossing Pixels : {int(np.sum(dog_out['binary_edges'] > 0)):6d} px (Max ZC  = {np.max(dog_out['zc_strength']):.4f})")

    if args.save_dir and HAS_CV2:
        os.makedirs(args.save_dir, exist_ok=True)
        cv2.imwrite(os.path.join(args.save_dir, "0_input_grayscale.png"), img)
        cv2.imwrite(os.path.join(args.save_dir, "1_sobel_edges.png"), sobel_out["binary_edges"])
        cv2.imwrite(os.path.join(args.save_dir, "2_prewitt_edges.png"), prewitt_out["binary_edges"])
        cv2.imwrite(os.path.join(args.save_dir, "3_kirsch_edges.png"), kirsch_out["binary_edges"])
        cv2.imwrite(os.path.join(args.save_dir, "4_log_edges.png"), log_out["binary_edges"])
        cv2.imwrite(os.path.join(args.save_dir, "5_dog_edges.png"), dog_out["binary_edges"])
        print(f"\nSaved all segmented output images to directory: {os.path.abspath(args.save_dir)}")


if __name__ == "__main__":
    main()