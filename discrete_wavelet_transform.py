import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg


def rgb_to_grey(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img.astype(np.uint8)
    if img.shape[2] == 4:
        img = img[:, :, :3]
    return (0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]).astype(np.uint8)


def _pad_to_even(img: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    h, w = img.shape[:2]
    pad_h = h % 2
    pad_w = w % 2
    if pad_h > 0 or pad_w > 0:
        padded = np.pad(img, ((0, pad_h), (0, pad_w)), mode="edge")
        return padded, (h, w)
    return img, (h, w)


def dwt1d_haar_rows(mat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sqrt2 = np.sqrt(2.0)
    low = (mat[:, 0::2] + mat[:, 1::2]) / sqrt2
    high = (mat[:, 0::2] - mat[:, 1::2]) / sqrt2
    return low, high


def idwt1d_haar_rows(low: np.ndarray, high: np.ndarray) -> np.ndarray:
    sqrt2 = np.sqrt(2.0)
    h, w_half = low.shape
    reconstructed = np.zeros((h, w_half * 2), dtype=np.float64)
    reconstructed[:, 0::2] = (low + high) / sqrt2
    reconstructed[:, 1::2] = (low - high) / sqrt2
    return reconstructed


def dwt1d_haar_cols(mat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sqrt2 = np.sqrt(2.0)
    low = (mat[0::2, :] + mat[1::2, :]) / sqrt2
    high = (mat[0::2, :] - mat[1::2, :]) / sqrt2
    return low, high


def idwt1d_haar_cols(low: np.ndarray, high: np.ndarray) -> np.ndarray:
    sqrt2 = np.sqrt(2.0)
    h_half, w = low.shape
    reconstructed = np.zeros((h_half * 2, w), dtype=np.float64)
    reconstructed[0::2, :] = (low + high) / sqrt2
    reconstructed[1::2, :] = (low - high) / sqrt2
    return reconstructed


def dwt2d_haar(image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, tuple[int, int]]:
    img_f = image.astype(np.float64)
    img_padded, orig_shape = _pad_to_even(img_f)

    l_row, h_row = dwt1d_haar_rows(img_padded)
    ll, lh = dwt1d_haar_cols(l_row)
    hl, hh = dwt1d_haar_cols(h_row)

    return ll, lh, hl, hh, orig_shape


def idwt2d_haar(
    ll: np.ndarray,
    lh: np.ndarray,
    hl: np.ndarray,
    hh: np.ndarray,
    orig_shape: tuple[int, int] | None = None,
) -> np.ndarray:
    l_row = idwt1d_haar_cols(ll, lh)
    h_row = idwt1d_haar_cols(hl, hh)
    reconstructed = idwt1d_haar_rows(l_row, h_row)

    if orig_shape is not None:
        reconstructed = reconstructed[:orig_shape[0], :orig_shape[1]]

    return reconstructed


def normalize_detail_subband(subband: np.ndarray) -> np.ndarray:
    abs_sub = np.abs(subband)
    max_val = np.max(abs_sub)
    if max_val > 1e-8:
        norm = (abs_sub / max_val) * 255.0
    else:
        norm = np.zeros_like(abs_sub)
    return np.clip(norm, 0, 255).astype(np.uint8)


def normalize_approx_subband(ll: np.ndarray) -> np.ndarray:
    norm = ll / 2.0
    return np.clip(norm, 0, 255).astype(np.uint8)


def build_1level_grid(
    ll: np.ndarray,
    lh: np.ndarray,
    hl: np.ndarray,
    hh: np.ndarray,
    border_color: int = 128,
    border_width: int = 2,
) -> np.ndarray:
    ll_u = normalize_approx_subband(ll)
    hl_u = normalize_detail_subband(hl)
    lh_u = normalize_detail_subband(lh)
    hh_u = normalize_detail_subband(hh)

    sep_h = np.full((border_width, ll_u.shape[1]), border_color, dtype=np.uint8)
    left_col = np.vstack((ll_u, sep_h, lh_u))
    right_col = np.vstack((hl_u, sep_h, hh_u))

    sep_v = np.full((left_col.shape[0], border_width), border_color, dtype=np.uint8)
    return np.hstack((left_col, sep_v, right_col))


def build_2level_grid(
    image: np.ndarray,
    border_color: int = 128,
    border_width: int = 2,
) -> tuple[np.ndarray, dict]:
    ll1, lh1, hl1, hh1, orig_shape1 = dwt2d_haar(image)
    ll2, lh2, hl2, hh2, orig_shape2 = dwt2d_haar(ll1)

    grid_lvl2 = build_1level_grid(ll2, lh2, hl2, hh2, border_color, border_width)

    hl1_u = normalize_detail_subband(hl1)
    lh1_u = normalize_detail_subband(lh1)
    hh1_u = normalize_detail_subband(hh1)

    h2, w2 = grid_lvl2.shape
    if hl1_u.shape[0] != h2:
        grid_lvl2 = np.pad(grid_lvl2, ((0, max(0, hl1_u.shape[0] - h2)), (0, 0)), mode="edge")[:hl1_u.shape[0], :]
    if hl1_u.shape[1] != w2:
        grid_lvl2 = np.pad(grid_lvl2, ((0, 0), (0, max(0, hl1_u.shape[1] - w2))), mode="edge")[:, :hl1_u.shape[1]]

    sep_h = np.full((border_width, grid_lvl2.shape[1]), border_color, dtype=np.uint8)
    left_col = np.vstack((grid_lvl2, sep_h, lh1_u))
    right_col = np.vstack((hl1_u, sep_h, hh1_u))

    sep_v = np.full((left_col.shape[0], border_width), border_color, dtype=np.uint8)
    grid_2level = np.hstack((left_col, sep_v, right_col))

    subbands_data = {
        "level1": {"LL": ll1, "LH": lh1, "HL": hl1, "HH": hh1, "shape": orig_shape1},
        "level2": {"LL": ll2, "LH": lh2, "HL": hl2, "HH": hh2, "shape": orig_shape2},
    }
    return grid_2level, subbands_data


def generate_comparison_figure(
    original_grey: np.ndarray,
    grid_1level: np.ndarray,
    grid_2level: np.ndarray,
    reconstructed_grey: np.ndarray,
    mse: float,
    max_err: float,
) -> Image.Image:
    fig, axes = plt.subplots(2, 2, figsize=(11, 10), facecolor="#1e1e2e")
    fig.suptitle(
        "Discrete Wavelet Transform (2D Haar DWT & IDWT)",
        fontsize=16,
        fontweight="bold",
        color="#cdd6f4",
        y=0.96,
    )

    panels = [
        (axes[0, 0], original_grey, "Original Image (Greyscale)", "Spatial domain"),
        (axes[0, 1], grid_1level, "1-Level DWT Subbands (2×2)", "Top-L: LL (Approx) | Top-R: HL (Vertical)\nBot-L: LH (Horiz)   | Bot-R: HH (Diagonal)"),
        (axes[1, 0], grid_2level, "2-Level Multiresolution DWT", "LL decomposed recursively into LL2, HL2, LH2, HH2"),
        (axes[1, 1], reconstructed_grey, "IDWT Reconstructed Image", f"Exact mathematical inverse\nMSE: {mse:.2e} | Max Diff: {max_err:.1f}"),
    ]

    for ax, data, title, subtitle in panels:
        ax.set_facecolor("#181825")
        ax.imshow(data, cmap="gray", vmin=0, vmax=255)
        ax.set_title(title, fontsize=12, fontweight="semibold", color="#89b4fa", pad=8)
        ax.set_xlabel(subtitle, fontsize=9, color="#a6adc8", labelpad=6)
        ax.tick_params(colors="#6c7086", labelsize=8)
        for spine in ax.spines.values():
            spine.set_color("#45475a")

    plt.tight_layout(rect=[0, 0.03, 1, 0.93])

    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    rgba = np.asarray(canvas.buffer_rgba())
    result_img = Image.fromarray(rgba[:, :, :3])
    plt.close(fig)
    return result_img


def process_haar_dwt(img_rgb: np.ndarray) -> dict:
    grey = rgb_to_grey(img_rgb)
    ll, lh, hl, hh, orig_shape = dwt2d_haar(grey)
    recon = idwt2d_haar(ll, lh, hl, hh, orig_shape=orig_shape)
    recon_uint8 = np.clip(np.round(recon), 0, 255).astype(np.uint8)

    mse_float = float(np.mean((grey.astype(np.float64) - recon) ** 2))
    max_abs_float = float(np.max(np.abs(grey.astype(np.float64) - recon)))

    grid_1level = build_1level_grid(ll, lh, hl, hh)
    grid_2level, _ = build_2level_grid(grey)

    fig_overview = generate_comparison_figure(grey, grid_1level, grid_2level, recon_uint8, mse_float, max_abs_float)

    return {
        "original_grey": grey,
        "grid_1level": grid_1level,
        "grid_2level": grid_2level,
        "reconstructed": recon_uint8,
        "overview_fig": fig_overview,
        "mse": mse_float,
        "max_err": max_abs_float,
    }