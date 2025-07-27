import cv2
import numpy as np
from numpy.fft import fft2, ifft2
import matplotlib.pyplot as plt

# Define globals
TEMPLATE_BANK_SIZE = 5
NCC_THRESHOLD = 0.5
template_bank = []

def masked_ncc_fft(search_image, template, mask):
    """
    Perform masked normalized cross-correlation using FFT.

    Args:
        search_image: Grayscale search region (H, W)
        template: Grayscale template (h, w)
        mask: Binary mask (same shape as template), 0=ignore, 1=valid
        visualize: If True, shows FFT magnitude spectra

    Returns:
        ncc_map: Valid NCC map (H - h + 1, W - w + 1)
        max_val: Max NCC score
        max_loc: Location (x, y) of max match
    """
    I = search_image.astype(np.float32)
    T = template.astype(np.float32)
    M = mask.astype(np.float32)

    H, W = I.shape
    h, w = T.shape

    # Pad template and mask to full image size
    TM = T * M
    TM_padded = np.zeros_like(I)
    M_padded = np.zeros_like(I)
    TM_padded[:h, :w] = TM
    M_padded[:h, :w] = M

    # FFTs
    fft_I     = fft2(I)
    fft_TM    = fft2(TM_padded)
    fft_M     = fft2(M_padded)
    fft_I2    = fft2(I ** 2)

    # Numerator and local sums
    num    = ifft2(fft_I * np.conj(fft_TM)).real
    sum_I  = ifft2(fft_I * np.conj(fft_M)).real
    sum_I2 = ifft2(fft_I2 * np.conj(fft_M)).real
    sum_M  = ifft2(fft_M * np.conj(fft_M)).real

    sum_TM  = np.sum(TM)
    sum_TM2 = np.sum(TM ** 2)

    denom = (sum_TM2 - (sum_TM ** 2) / (sum_M + 1e-8)) * (sum_I2 - (sum_I ** 2) / (sum_M + 1e-8))
    denom = np.maximum(denom, 0)
    denom = np.sqrt(denom)

    with np.errstate(divide='ignore', invalid='ignore'):
        ncc_map = (num - (sum_I * sum_TM / (sum_M + 1e-8))) / denom
        ncc_map[sum_M < 1] = 0

    valid_h = H - h + 1
    valid_w = W - w + 1
    ncc_valid = ncc_map[:valid_h, :valid_w]

    _, max_val, _, max_loc = cv2.minMaxLoc(ncc_valid)

    return ncc_valid, max_val, max_loc

def get_patch(frame_gray, template_coords, padding=10):
    x, y, w, h = template_coords
    x_start = max(0, x - padding)
    y_start = max(0, y - padding)
    x_end = min(frame_gray.shape[1], x + w + padding)
    y_end = min(frame_gray.shape[0], y + h + padding)

    patch = frame_gray[y_start:y_end, x_start:x_end]
    patch_loc = (x_start, y_start)  # top-left of patch

    return patch, patch_loc

def display_patch(patch, frame_idx, padding):
    plt.figure(figsize=(4, 4))
    plt.imshow(patch, cmap='gray')
    plt.title(f"Patch (Frame {frame_idx}, Padding {padding})")
    plt.axis('off')
    plt.show()

def find_best_match(search_patch):
    best_score = -1
    best_box = None
    boxes = []
    scores = []

    for template_img, template_mask in template_bank:
        h, w = template_img.shape
        H, W = search_patch.shape

        if H < h or W < w:
            continue  # Skip if template larger than search patch

        # Use masked NCC in FFT
        _, score, loc = masked_ncc_fft(search_patch, template_img, template_mask)

        if score > best_score:
            best_score = score
            best_box = (*loc, w, h)

        boxes.append((*loc, w, h))
        scores.append(score)

    if best_box is None:
        best_box = (0, 0, 0, 0)

    return boxes, scores, best_score

def update_template_from_box(frame_gray, box):
    x, y, w, h = box
    new_template = frame_gray[y:y+h, x:x+w]
    _, new_mask = cv2.threshold(new_template, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return new_template, new_mask

def add_to_template_bank(template, mask, max_size=TEMPLATE_BANK_SIZE):
    global template_bank
    template_bank.append((template, mask))  # Now a tuple
    if len(template_bank) > max_size:
        template_bank.pop(0)

def apply_nms(boxes, scores, threshold, nms_thresh):
    if not boxes:
        return []

    indices = cv2.dnn.NMSBoxes(boxes, scores, threshold, nms_thresh)
    if len(indices) == 0:
        return []

    # Flatten indices to list of integers
    return [i[0] if isinstance(i, (list, np.ndarray)) else i for i in indices]

def template_difference(template1, template2, threshold=0.2):
    if template1.shape != template2.shape:
        return True  # treat different shape as significantly different
    diff = np.mean((template1.astype(np.float32) - template2.astype(np.float32)) ** 2)
    norm = np.mean(template1.astype(np.float32) ** 2)
    nmse = diff / (norm + 1e-8)
    return nmse > threshold

#================================================#
#======= FOR TEMPLATE BANK VISUALIZATION ========#
#================================================#

def get_template_index(template, template_bank):
    for i, (t, _) in enumerate(template_bank):
        if np.array_equal(t, template):
            return i
    return None

def visualize_template_bank(template_bank, current_index=None, max_per_row=5, figsize=(15, 5), cmap='gray'):
    n = len(template_bank)
    if n == 0:
        print("⚠️ Template bank is empty.")
        return

    cols = min(n, max_per_row)
    rows = (n + cols - 1) // cols

    plt.figure(figsize=figsize)
    for i, (template, mask) in enumerate(template_bank):
        ax = plt.subplot(rows, cols, i + 1)

        # Fix mask dtype and size
        if mask.dtype != np.uint8:
            mask = mask.astype(np.uint8)
        if mask.shape != template.shape:
            mask = cv2.resize(mask, (template.shape[1], template.shape[0]), interpolation=cv2.INTER_NEAREST)

        masked = cv2.bitwise_and(template, template, mask=mask)
        overlay = np.stack([masked]*3, axis=-1)
        overlay[mask == 0] = [255, 0, 0]  # red for masked out
        ax.imshow(overlay)
        title = f"Template {i}"
        if i == current_index:
            title += " (Current)"
            ax.set_title(title, color='red')
            for spine in ax.spines.values():
                spine.set_edgecolor('red')
                spine.set_linewidth(2)
        else:
            ax.set_title(title)
        plt.axis('off')
    plt.tight_layout()
    plt.show()
