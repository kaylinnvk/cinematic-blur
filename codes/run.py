from vos_package.graphcut import graphcut_segmentation
from vos_package.blur import blur_background
import cv2
import os
import sys

def get_input_image():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.abspath(os.path.join(base_dir, "../images/clay.png"))
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image: {image_path}")
        sys.exit(1)
    return image

def show_resized(window_name, image, max_size=1000):
    h, w = image.shape[:2]
    scale = min(max_size / h, max_size / w, 1.0)
    resized = cv2.resize(image, (int(w * scale), int(h * scale)))
    cv2.imshow(window_name, resized)

def main():
    image = get_input_image()

    _, mask = graphcut_segmentation(image)
    output = blur_background(image, mask)

    show_resized('Blurred Background', output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()