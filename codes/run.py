from vos_package.graphcut import graphcut_segmentation
from vos_package.mask_propagation import dynamic_template_matching
import cv2
import os
import cProfile
import pstats

def get_first_frame(video_path):
    """
    Reads and returns the first frame of the video at video_path.

    Args: 
        video_path (str): The video path
        
    Returns:
        np.ndarray: The first frame as a BGR image, or None if failed.
    """
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print(f"Failed to read first frame from {video_path}")
        return None
    return frame

def show_resized(window_name, image, max_size=1000):
    h, w = image.shape[:2]
    scale = min(max_size / h, max_size / w, 1.0)
    resized = cv2.resize(image, (int(w * scale), int(h * scale)))
    cv2.imshow(window_name, resized)

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    video_path = os.path.abspath(os.path.join(base_dir, "../data/personal/videos/paperplane.mp4"))
    output_path = os.path.abspath(os.path.join(base_dir, "../outputs/personal/blurred/blurred_paperplane.mp4"))

    first_frame = get_first_frame(video_path)
    if first_frame is None:
        print("Aborting: Could not read the first frame.")
        return

    segmented, mask = graphcut_segmentation(first_frame)
    if segmented is None or mask is None:
        print("Aborting: Graphcut segmentation failed.")
        return
    
    profiler = cProfile.Profile()
    profiler.enable()

    dynamic_template_matching(video_path, segmented, mask, output_path)

    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('cumtime')
    stats.print_stats(20)

if __name__ == "__main__":
    main()