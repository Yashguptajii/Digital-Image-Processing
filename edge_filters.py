import numpy as np
from PIL import Image


def rgb_to_grey(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img.astype(np.uint8)
    if img.ndim == 3 and img.shape[2] >= 3:
        return (0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]).astype(np.uint8)
    return img.astype(np.uint8)


def conv2d_scratch(image: np.ndarray, kernel: np.ndarray, pad_mode: str = "edge") -> np.ndarray:
    img_f = image.astype(np.float64)
    k_h, k_w = kernel.shape
    pad_h = k_h // 2
    pad_w = k_w // 2

    k_rot = np.flipud(np.fliplr(kernel))
    padded = np.pad(img_f, ((pad_h, pad_h), (pad_w, pad_w)), mode=pad_mode)

    h, w = img_f.shape
    output = np.zeros((h, w), dtype=np.float64)

    for i in range(k_h):
        for j in range(k_w):
            weight = k_rot[i, j]
            if weight != 0.0:
                output += weight * padded[i : i + h, j : j + w]

    return output


def contrast_stretch_uint8(arr: np.ndarray) -> np.ndarray:
    min_v = np.min(arr)
    max_v = np.max(arr)
    if max_v - min_v > 1e-8:
        norm = (arr - min_v) / (max_v - min_v) * 255.0
    else:
        norm = np.zeros_like(arr)
    return np.clip(np.round(norm), 0, 255).astype(np.uint8)


def forward_difference(image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    img_f = image.astype(np.float64)
    h, w = img_f.shape
    padded = np.pad(img_f, ((0, 1), (0, 1)), mode="edge")

    gx = padded[:h, 1 : w + 1] - padded[:h, :w]
    gy = padded[1 : h + 1, :w] - padded[:h, :w]
    mag = np.sqrt(gx**2 + gy**2)
    return gx, gy, contrast_stretch_uint8(mag)


def backward_difference(image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    img_f = image.astype(np.float64)
    h, w = img_f.shape
    padded = np.pad(img_f, ((1, 0), (1, 0)), mode="edge")

    gx = padded[1 : h + 1, 1 : w + 1] - padded[1 : h + 1, :w]
    gy = padded[1 : h + 1, 1 : w + 1] - padded[:h, 1 : w + 1]
    mag = np.sqrt(gx**2 + gy**2)
    return gx, gy, contrast_stretch_uint8(mag)


def central_difference(image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    img_f = image.astype(np.float64)
    h, w = img_f.shape
    padded = np.pad(img_f, ((1, 1), (1, 1)), mode="edge")

    gx = (padded[1 : h + 1, 2 : w + 2] - padded[1 : h + 1, :w]) / 2.0
    gy = (padded[2 : h + 2, 1 : w + 1] - padded[:h, 1 : w + 1]) / 2.0
    mag = np.sqrt(gx**2 + gy**2)
    return gx, gy, contrast_stretch_uint8(mag)


def generate_gaussian_kernel_2d(sigma: float, size: int | None = None) -> np.ndarray:
    if size is None:
        radius = int(np.ceil(3.0 * sigma))
        size = 2 * radius + 1
    else:
        radius = size // 2

    y, x = np.mgrid[-radius : radius + 1, -radius : radius + 1]
    kernel = np.exp(-(x**2 + y**2) / (2.0 * sigma**2))
    kernel_sum = np.sum(kernel)
    if kernel_sum > 0:
        kernel /= kernel_sum
    return kernel


def generate_log_kernel(sigma: float, size: int | None = None) -> np.ndarray:
    if size is None:
        radius = int(np.ceil(3.0 * sigma))
        size = 2 * radius + 1
    else:
        radius = size // 2

    y, x = np.mgrid[-radius : radius + 1, -radius : radius + 1]
    r2 = (x**2 + y**2).astype(np.float64)
    sigma2 = sigma**2
    sigma4 = sigma**4

    log_kernel = -(1.0 / (np.pi * sigma4)) * (1.0 - r2 / (2.0 * sigma2)) * np.exp(-r2 / (2.0 * sigma2))
    return log_kernel - np.mean(log_kernel)


def detect_zero_crossings(response: np.ndarray, threshold_ratio: float = 0.04) -> np.ndarray:
    max_amp = np.max(np.abs(response))
    if max_amp < 1e-8:
        return np.zeros_like(response, dtype=np.uint8)

    tau = threshold_ratio * max_amp
    p = np.pad(response, 1, mode="edge")
    c = p[1:-1, 1:-1]

    left, right = p[1:-1, :-2], p[1:-1, 2:]
    h_cross = ((left * right < 0) | (c * left < 0) | (c * right < 0)) & (np.abs(left - right) > tau)

    up, down = p[:-2, 1:-1], p[2:, 1:-1]
    v_cross = ((up * down < 0) | (c * up < 0) | (c * down < 0)) & (np.abs(up - down) > tau)

    ul, br = p[:-2, :-2], p[2:, 2:]
    d1_cross = ((ul * br < 0) | (c * ul < 0) | (c * br < 0)) & (np.abs(ul - br) > tau)

    ur, bl = p[:-2, 2:], p[2:, :-2]
    d2_cross = ((ur * bl < 0) | (c * ur < 0) | (c * bl < 0)) & (np.abs(ur - bl) > tau)

    return ((h_cross | v_cross | d1_cross | d2_cross).astype(np.uint8) * 255)


def laplacian_of_gaussian(
    image: np.ndarray,
    sigma: float = 1.4,
    threshold_ratio: float = 0.04,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    kernel = generate_log_kernel(sigma)
    response = conv2d_scratch(image, kernel, pad_mode="edge")
    return response, contrast_stretch_uint8(response), detect_zero_crossings(response, threshold_ratio)


def difference_of_gaussians(
    image: np.ndarray,
    sigma1: float = 1.0,
    sigma2: float = 1.6,
    threshold_ratio: float = 0.04,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    k1 = generate_gaussian_kernel_2d(sigma1)
    k2 = generate_gaussian_kernel_2d(sigma2)

    g1 = conv2d_scratch(image, k1, pad_mode="edge")
    g2 = conv2d_scratch(image, k2, pad_mode="edge")

    dog_response = g1 - g2
    return dog_response, contrast_stretch_uint8(dog_response), detect_zero_crossings(dog_response, threshold_ratio)


def to_3ch(img_2d: np.ndarray) -> np.ndarray:
    if img_2d.ndim == 3:
        return img_2d
    return np.stack([img_2d, img_2d, img_2d], axis=-1)


def build_mosaic_grid(
    images: list[np.ndarray],
    rows: int,
    cols: int,
    border_width: int = 2,
    border_color: int = 100,
) -> np.ndarray:
    h, w = images[0].shape[:2]
    std_images = []
    for im in images:
        if im.shape[:2] != (h, w):
            pil_im = Image.fromarray(im).resize((w, h), Image.Resampling.BILINEAR)
            im = np.array(pil_im)
        std_images.append(to_3ch(im))

    total_slots = rows * cols
    while len(std_images) < total_slots:
        std_images.append(np.zeros((h, w, 3), dtype=np.uint8))

    row_blocks = []
    idx = 0
    for _ in range(rows):
        col_cells = []
        for c in range(cols):
            col_cells.append(std_images[idx])
            if c < cols - 1:
                col_cells.append(np.full((h, border_width, 3), border_color, dtype=np.uint8))
            idx += 1
        row_blocks.append(np.hstack(col_cells))

    grid_rows = []
    for r in range(rows):
        grid_rows.append(row_blocks[r])
        if r < rows - 1:
            grid_rows.append(np.full((border_width, row_blocks[r].shape[1], 3), border_color, dtype=np.uint8))

    return np.vstack(grid_rows)


def process_edge_detection(img_rgb: np.ndarray) -> dict:
    grey = rgb_to_grey(img_rgb)

    _, _, fwd_mag = forward_difference(grey)
    _, _, bwd_mag = backward_difference(grey)
    _, _, cen_mag = central_difference(grey)

    _, log_resp_u8, log_zc = laplacian_of_gaussian(grey, sigma=1.4, threshold_ratio=0.04)
    _, dog_resp_u8, dog_zc = difference_of_gaussians(grey, sigma1=1.0, sigma2=1.6, threshold_ratio=0.04)

    grid_1st = build_mosaic_grid([grey, fwd_mag, bwd_mag, cen_mag], rows=1, cols=4)
    grid_2nd = build_mosaic_grid([log_resp_u8, log_zc, dog_resp_u8, dog_zc], rows=2, cols=2)
    grid_master = build_mosaic_grid(
        [grey, fwd_mag, bwd_mag, cen_mag, log_resp_u8, log_zc, dog_resp_u8, dog_zc],
        rows=2,
        cols=4,
    )

    return {
        "grey": grey,
        "forward": fwd_mag,
        "backward": bwd_mag,
        "central": cen_mag,
        "log_response": log_resp_u8,
        "log_zc": log_zc,
        "dog_response": dog_resp_u8,
        "dog_zc": dog_zc,
        "grid_1st": grid_1st,
        "grid_2nd": grid_2nd,
        "master_grid": grid_master,
    }