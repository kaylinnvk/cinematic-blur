import cv2
import numpy as np

def blur_background(image: np.ndarray, mask: np.ndarray, ksize: tuple = (33, 33), sigma: int = 0) -> np.ndarray:
    """
    Blurs the background of the image while keeping the foreground (mask) sharp.

    Args:
        image (np.ndarray): RGB image of shape (H, W, 3).
        mask (np.ndarray): Binary mask of shape (H, W), foreground=1 or 255, background=0.
        ksize (tuple, optional): Kernel size for Gaussian blur. Defaults to (23, 23).
        sigma (int, optional): Standard deviation for Gaussian blur. Defaults to 0.

    Returns:
        np.ndarray: Result image with blurred background and sharp foreground.
    """
    # Ensure mask is binary and 3-channel
    mask_bin = (mask > 0).astype(np.uint8)
    mask_3c = np.repeat(mask_bin[:, :, None], 3, axis=2)

    blurred = cv2.GaussianBlur(image, ksize, sigma)
    combined = np.where(mask_3c == 1, image, blurred)

    return combined

def show_resized(window_name, image, max_size=1000):
    h, w = image.shape[:2]
    scale = min(max_size / h, max_size / w, 1.0)  # scale ≤ 1.0, only shrink if too big
    resized = cv2.resize(image, (int(w * scale), int(h * scale)))
    cv2.imshow(window_name, resized)

if __name__ == "__main__":
    import sys
    import os

    base_dir = os.path.dirname(os.path.abspath(__file__))  # directory where script is
    image_path = os.path.abspath(os.path.join(base_dir, "../../images/clay.png"))
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image: {image_path}")
        sys.exit(1)

    # Create a dummy mask: a white circle in center as foreground, rest background
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    h, w = mask.shape
    cv2.circle(mask, (w//2, h//2), min(h, w)//4, 255, -1)  # white circle mask

    # Call blur function
    result = blur_background(image, mask)

    # Show original and result side-by-side
    combined_display = np.hstack([image, result])
    show_resized("Original (left) vs Blurred Background (right)", combined_display, max_size=800)

    cv2.waitKey(0)
    cv2.destroyAllWindows()