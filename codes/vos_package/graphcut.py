import cv2
import numpy as np
from skimage.segmentation import slic
from scipy.stats import norm
import maxflow
import time

# Define globals
drawing = False
scribble_mask = []
current_label = 1  # 1 = FG (red), 2 = BG (blue)
curve_points = []
scale_x = scale_y = 1.0
brush_radius = 3

def reset_scribble_mask(shape: tuple) -> None:
    '''
    Reset the global scribble mask to a zero matrix of given shape.
    
    Args:
        shape (tuple): The shape (height, width) to initialize the scribble mask.
    
    Returns:
        None
    '''
    global scribble_mask
    scribble_mask = np.zeros(shape, dtype=np.uint8)


def scribble_seeds(event: int, x: int, y: int, flags, param) -> None:
    '''
    Mouse callback to draw scribbles for foreground and background seeds on the scribble mask.
    
    Args:
        event (int): The mouse event type (e.g., left button down, right button down).
        x (int): The x-coordinate of the mouse event.
        y (int): The y-coordinate of the mouse event.
        flags: Additional flags (not used here).
        param: Additional parameters (not used here).
    
    Returns:
        None
    '''
    global drawing, current_label, scale_x, scale_y

    img_x = int(x * scale_x)
    img_y = int(y * scale_y)

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        current_label = 1  # Foreground
    elif event == cv2.EVENT_RBUTTONDOWN:
        drawing = True
        current_label = 2  # Background
    elif event in [cv2.EVENT_LBUTTONUP, cv2.EVENT_RBUTTONUP]:
        drawing = False
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        if 0 <= img_x < scribble_mask.shape[1] and 0 <= img_y < scribble_mask.shape[0]:
            cv2.circle(scribble_mask, (img_x, img_y), 5, current_label, -1)


def get_user_seeds(image: np.ndarray, segments: np.ndarray, masked_image: np.ndarray = None, max_display_size: int = 800, alpha: float = 0.5):
    '''
    Display the image for the user to draw scribbles indicating foreground and background seeds.
    Collect and return the sets of segment IDs corresponding to user-marked foreground and background.
    
    Args:
        image (np.ndarray): The original image.
        segments (np.ndarray): The superpixel segments of the image.
        masked_image (np.ndarray, optional): An optional masked image overlay for visualization.
        max_display_size (int, optional): Max dimension to scale the display window. Defaults to 800.
        alpha (float, optional): Alpha blending factor for overlay. Defaults to 0.5.
    
    Returns:
        tuple:
            fg_seeds (set): Set of segment IDs marked as foreground by user.
            bg_seeds (set): Set of segment IDs marked as background by user.
            overlay (np.ndarray): The final overlay image shown to user with scribbles.
    '''
    global scribble_mask, scale_x, scale_y, brush_radius

    segments = segments.astype(np.int32)
    scribble_mask = np.zeros(image.shape[:2], dtype=np.uint8)

    # Calculate display scaling (for display only)
    h, w = image.shape[:2]
    scale = min(max_display_size / h, max_display_size / w, 1.0)
    display_size = (int(w * scale), int(h * scale))
    scale_x = w / display_size[0]
    scale_y = h / display_size[1]

    brush_radius = max(1, int(3 * (h / 1080) ** 0.5))

    cv2.namedWindow("Scribble on Image", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Scribble on Image", *display_size)
    cv2.setMouseCallback("Scribble on Image", scribble_seeds)

    while True:
        if masked_image is not None:
            img_float = image.astype(np.float32)
            masked_float = masked_image.astype(np.float32)
            overlay = cv2.addWeighted(img_float, alpha, masked_float, 1 - alpha, 0).astype(np.uint8)
        else:
            overlay = image.copy()

        # Draw scribbles on overlay
        overlay[scribble_mask == 1] = [0, 0, 255]  # Red = foreground
        overlay[scribble_mask == 2] = [255, 0, 0]  # Blue = background

        overlay_resized = cv2.resize(overlay, display_size, interpolation=cv2.INTER_NEAREST)
        cv2.imshow("Scribble on Image", overlay_resized)

        key = cv2.waitKey(1)
        if key == 13:  # Enter key to finish
            break

    cv2.destroyAllWindows()

    # Get segment IDs marked as foreground and background seeds
    fg_seeds = set(np.unique(segments[scribble_mask == 1]))
    bg_seeds = set(np.unique(segments[scribble_mask == 2]))
    fg_seeds.discard(0)
    bg_seeds.discard(0)

    return fg_seeds, bg_seeds, overlay

def superpixel_segmentation(image: np.ndarray) -> np.ndarray:
    '''
    Apply low level segmentation to reduce computational costs. 
    Use superpixel segmentation instead of EDISON mean shift for better 
    implementation in Python.
    
    Args:
        image (np.ndarray): Input image to segment.
    
    Returns:
        np.ndarray: Segment labels for each pixel.
    '''
    return slic(image, n_segments=1300, compactness=10)


def compute_gaussian_likelihoods(image: np.ndarray, segments: np.ndarray, fg_seeds: set, bg_seeds: set):
    '''
    Computes per-pixel probability of belonging to foreground or background
    using a fixed Gaussian model from user scribbles.
    Calculates the likelihoods using SciPy's norm function.
    Used for the data term estimation.
    
    Args:
        image (np.ndarray): Input image.
        segments (np.ndarray): Segment labels for each pixel.
        fg_seeds (set): Set of segment IDs marked as foreground.
        bg_seeds (set): Set of segment IDs marked as background.
    
    Returns:
        tuple: Two dictionaries mapping segment IDs to their foreground and background likelihoods.
    '''
    fg_pixels = np.vstack([
        image[segments == s] for s in fg_seeds if np.any(segments == s)
    ])
    bg_pixels = np.vstack([
        image[segments == s] for s in bg_seeds if np.any(segments == s)
    ])

    print("[compute_GMM_likelihood] Computing likelihoods...")

    # Compute mean and std for each channel
    fg_mean, fg_std = fg_pixels.mean(axis=0), fg_pixels.std(axis=0) + 1e-5
    bg_mean, bg_std = bg_pixels.mean(axis=0), bg_pixels.std(axis=0) + 1e-5

    # Now compute likelihoods for each segment
    segment_ids = np.unique(segments)
    fg_likelihoods, bg_likelihoods = {}, {}

    for s in segment_ids:
        pix = image[segments == s]
        fg_prob = norm.pdf(pix, loc=fg_mean, scale=fg_std).prod(axis=1).mean()
        bg_prob = norm.pdf(pix, loc=bg_mean, scale=bg_std).prod(axis=1).mean()
        fg_likelihoods[s] = fg_prob
        bg_likelihoods[s] = bg_prob

    print("[compute_GMM_likelihood] Done.")
    return fg_likelihoods, bg_likelihoods


def compute_segment_means(image: np.ndarray, segments: np.ndarray, n_segments: int) -> np.ndarray:
    '''
    Compute mean color per segment for smoothness (N-links)
    
    Args:
        image (np.ndarray): Input image.
        segments (np.ndarray): Segment labels per pixel.
        n_segments (int): Number of segments.
    
    Returns:
        np.ndarray: Array of mean RGB colors per segment.
    '''
    segment_means = np.zeros((n_segments, 3))
    for i in range(n_segments):
        mask = (segments == i)
        if np.any(mask):
            segment_means[i] = np.mean(image[mask], axis=0)
    return segment_means


def build_RAG(segments: np.ndarray) -> set:
    '''
    Build a set of edges (segment pairs that are neighbors).
    
    Args:
        segments (np.ndarray): Segment label map.
    
    Returns:
        set: Set of tuples (u, v) where u and v are neighboring segment IDs.
    '''
    H, W = segments.shape
    edges = set()
    for y in range(H - 1):
        for x in range(W - 1):
            node = segments[y, x]
            # Only consider neighboring pixels to the right and below
            # The others are already considered in earlier scans
            node_right = segments[y, x + 1]
            node_below = segments[y + 1, x]
            # Only take nodes of differently-labeled segments
            if node != node_right:
                edges.add(tuple(sorted((node, node_right))))
            if node != node_below:
                edges.add(tuple(sorted((node, node_below))))
    print(f"[build_RAG] Built RAG with {len(edges)} edges.")
    return edges


def add_n_links(graph, edges: set, segment_means: np.ndarray, node_ids, sigma_sq: float = 1000, lambda_val: float = 50):
    '''
    Add n-link weights (smoothness term) based on color similarity between neighboring segments.

    Args:
        graph: PyMaxflow graph object.
        edges (set): Set of (u, v) tuples, each representing an edge between segment IDs.
        segment_means (np.ndarray): Array of mean RGB colors for each segment.
        node_ids: List or array of graph node indices, one per segment.
        sigma_sq (float, optional): Variance parameter for Gaussian weighting. Defaults to 1000.
        lambda_val (float, optional): Weight scaling factor for smoothness term. Defaults to 50.
    
    Returns:
        graph: The graph with added n-links.
    '''
    for u, v in edges:
        color_diff = np.linalg.norm(segment_means[u] - segment_means[v])
        weight = lambda_val * np.exp(-color_diff**2 / (2 * sigma_sq))
        weight = max(1e-3, weight)  # prevent weight = 0
        graph.add_edge(node_ids[u], node_ids[v], weight, weight)
    return graph


def add_t_links(graph, fg_seeds: set, bg_seeds: set, fg_likelihoods: dict, bg_likelihoods: dict, node_ids):
    """
    Add terminal links (t-links) for source/sink connections in the graph cut.

    Args:
        graph: PyMaxflow Graph object.
        fg_seeds (set): Set of segment indices labeled as foreground.
        bg_seeds (set): Set of segment indices labeled as background.
        fg_likelihoods (dict): Per-segment foreground likelihoods.
        bg_likelihoods (dict): Per-segment background likelihoods.
        node_ids: List or array of node indices returned by graph.add_nodes().
    
    Returns:
        graph: The graph with added terminal links.
    """
    epsilon = 1e-6  # minimum likelihood to avoid log(0)
    for s in fg_likelihoods:
        if s in fg_seeds:
            graph.add_tedge(node_ids[s], 1000, 0)
        elif s in bg_seeds:
            graph.add_tedge(node_ids[s], 0, 1000)
        else:
            fg_prob = max(fg_likelihoods.get(s, epsilon), epsilon)
            bg_prob = max(bg_likelihoods.get(s, epsilon), epsilon)
            fg_cost = -np.log(fg_prob)
            bg_cost = -np.log(bg_prob)
            graph.add_tedge(node_ids[s], fg_cost, bg_cost)
    return graph

def run_graph_cut(image: np.ndarray, segments: np.ndarray, fg_seeds: set, bg_seeds: set) -> np.ndarray:
    '''
    Perform graph cut using the Boykov-Kolmogorov Algorithm.
    
    Args:
        image (np.ndarray): Input image.
        segments (np.ndarray): Superpixel segment labels for the image.
        fg_seeds (set): Set of segment IDs labeled as foreground by user.
        bg_seeds (set): Set of segment IDs labeled as background by user.
    
    Returns:
        np.ndarray: Binary mask where 1 indicates foreground and 0 background.
    '''
    n_segments = segments.max() + 1
    if not fg_seeds:
        raise ValueError(f"No valid foreground seed segments found in the segmentation. Please select at least one foreground segment (left-click).")
    if not bg_seeds:
        raise ValueError(f"No valid background seed segments found in the segmentation. Please select at least one background segment (right-click).")

    fg_likelihoods, bg_likelihoods = compute_gaussian_likelihoods(image, segments, fg_seeds, bg_seeds)
    segment_means = compute_segment_means(image, segments, n_segments)
    edges = build_RAG(segments)
    
    graph = maxflow.Graph[float](n_segments, len(edges))
    node_ids = graph.add_nodes(n_segments)

    graph = add_n_links(graph, edges, segment_means, node_ids)
    graph = add_t_links(graph, fg_seeds, bg_seeds, fg_likelihoods, bg_likelihoods, node_ids)

    print("[graph_cut_segmentation] Performing maxflow...")
    flow = graph.maxflow()
    print(f"[graph_cut_segmentation] Done. Maxflow: {flow:.4f}")

    # Extract segmentation mask: 1 if foreground, 0 if background
    segment_labels = np.array([graph.get_segment(node_ids[s]) for s in range(n_segments)])
    mask = np.zeros_like(segments, dtype=np.uint8)
    for s in range(n_segments):
        mask[segments == s] = 1 if segment_labels[s] == 0 else 0  # 0: source (fg), 1: sink (bg)
    
    return mask

def prepare_segments(image: np.ndarray):
    '''
    Perform superpixel segmentation and get initial foreground/background seeds from user scribbles.
    
    Args:
        image (np.ndarray): Input image.
    
    Returns:
        tuple:
            segments (np.ndarray): Superpixel segment labels.
            initial_fg_seeds (set): Set of segment IDs labeled foreground by user.
            initial_bg_seeds (set): Set of segment IDs labeled background by user.
    '''
    print(f"[superpixel_segmentation] Start superpixel segmentation.")
    start_time = time.time()
    segments = superpixel_segmentation(image)
    end_time = time.time()
    elapsed = end_time - start_time
    print(f"[superpixel_segmentation] Function took {elapsed:.4f} seconds")
    print(f"[superpixel_segmentation] There are {len(np.unique(segments))} unique segments")

    reset_scribble_mask(image.shape[:2])
    print(f"[get_user_seeds] Scribble image to define foreground (RED - right-click; BLUE - left-click)")
    initial_fg_seeds, initial_bg_seeds, overlay = get_user_seeds(image, segments)
    print(f"[get_user_seeds] # of initial foreground seeds: {len(initial_fg_seeds)}")
    print(f"[get_user_seeds] # of initial background seeds: {len(initial_bg_seeds)}")
    
    return segments, initial_fg_seeds, initial_bg_seeds 


def foreground_segmentation(image: np.ndarray, segments: np.ndarray, initial_fg_seeds: set, initial_bg_seeds: set):
    '''
    Run graph cut segmentation to generate foreground mask and segmented image.
    
    Args:
        image (np.ndarray): Input image.
        segments (np.ndarray): Superpixel segment labels.
        initial_fg_seeds (set): Initial foreground seed segments.
        initial_bg_seeds (set): Initial background seed segments.
    
    Returns:
        tuple:
            segmented (np.ndarray): Image masked by foreground.
            mask (np.ndarray): Binary mask of foreground.
    '''
    start_time = time.time()
    mask = run_graph_cut(image, segments, initial_fg_seeds, initial_bg_seeds)
    end_time = time.time()
    elapsed = end_time - start_time
    print(f"Graph Cut Function took {elapsed:.4f} seconds.")
    
    segmented = image * mask[:, :, np.newaxis]

    return segmented, mask


def create_overlay(original: np.ndarray, mask: np.ndarray, alpha: float = 0.2) -> np.ndarray:
    '''
    Create an overlay image by blending the original image with black where mask is off.
    
    Args:
        original (np.ndarray): Original image.
        mask (np.ndarray): Binary mask of foreground.
        alpha (float, optional): Blending factor for original image. Defaults to 0.2.
    
    Returns:
        np.ndarray: Overlay image with masked regions blended.
    '''
    # Ensure mask is binary and 3-channel
    mask_bin = (mask > 0).astype(np.uint8)
    mask_3c = np.repeat(mask_bin[:, :, None], 3, axis=2)

    black = np.zeros_like(original)
    blended = cv2.addWeighted(original, alpha, black, 1 - alpha, 0)
    
    overlay = np.where(mask_3c == 1, original, blended)
    return overlay


def refine(original: np.ndarray, full_image: np.ndarray, full_mask: np.ndarray, segments_full: np.ndarray, all_fg_seeds: set, all_bg_seeds: set):
    '''
    Refine the segmentation mask with new user scribbles on an overlay of the current segmentation.
    
    Args:
        original (np.ndarray): Original image.
        full_image (np.ndarray): Currently segmented image.
        full_mask (np.ndarray): Current binary mask.
        segments_full (np.ndarray): Full segments of the image.
        all_fg_seeds (set): Current set of foreground seed segment IDs.
        all_bg_seeds (set): Current set of background seed segment IDs.
    
    Returns:
        tuple:
            refined_segmented (np.ndarray): Refined segmented image.
            refined_mask (np.ndarray): Refined binary mask.
            all_fg_seeds (set): Updated foreground seeds.
            all_bg_seeds (set): Updated background seeds.
    '''
    print("[refine] Refining Image...")

    reset_scribble_mask(original.shape[:2])
    overlay = create_overlay(original, full_mask)

    new_fg_seeds, new_bg_seeds, _ = get_user_seeds(overlay, segments_full)

    if len(new_fg_seeds) == 0 and len(new_bg_seeds) == 0:
        print("[refine] No new scribbles, skipping refinement.")
        return full_image * full_mask[:, :, np.newaxis], full_mask, all_fg_seeds, all_bg_seeds

    # Conflict resolution: remove segments that are redefined by user
    all_fg_seeds.difference_update(new_bg_seeds)
    all_bg_seeds.difference_update(new_fg_seeds)

    # Update seeds
    all_fg_seeds.update(new_fg_seeds)
    all_bg_seeds.update(new_bg_seeds)

    # Start with current mask's segments (preserve old FG)
    refined_mask = (full_mask > 0).astype(np.uint8)

    # Remove newly defined background
    for s in all_bg_seeds:
        refined_mask[segments_full == s] = 0

    # Add newly defined foreground
    for s in all_fg_seeds:
        refined_mask[segments_full == s] = 1

    refined_segmented = full_image * refined_mask[:, :, np.newaxis]
    return refined_segmented, refined_mask, all_fg_seeds, all_bg_seeds

def show_image_resized(window_name: str, image: np.ndarray, max_size: int = 800) -> None:
    """
    Displays an image in a window scaled to fit within max_size.
    
    Args:
        window_name (str): Name of the OpenCV window.
        image (np.ndarray): Image to show (H x W x 3 or H x W).
        max_size (int, optional): Max width or height of the displayed image. Defaults to 800.
    
    Returns:
        None
    """
    h, w = image.shape[:2]
    scale = min(max_size / h, max_size / w, 1.0)
    resized = cv2.resize(image, (int(w * scale), int(h * scale)))
    cv2.imshow(window_name, resized)


def graphcut_segmentation(image):
    '''
    Consists of the whole graph cut pipeline.
    
    Steps:
    - Reads input image.
    - Performs superpixel segmentation.
    - Gets user scribbles for foreground and background seeds.
    - Runs graph cut segmentation.
    - Displays initial segmentation.
    - Optionally allows iterative user refinement with updated scribbles.
    
    Returns:
        tuple: Final segmented image and mask.
    '''
    segments_full = superpixel_segmentation(image)

    reset_scribble_mask(image.shape[:2])
    print(f"[get_user_seeds] Scribble image to define foreground (RED - left-click; BLUE - right-click)")
    initial_fg_seeds, initial_bg_seeds, _ = get_user_seeds(image, segments_full)
    print(f"[get_user_seeds] # of initial foreground seeds: {len(initial_fg_seeds)}")
    print(f"[get_user_seeds] # of initial background seeds: {len(initial_bg_seeds)}")

    initial_segmented, initial_mask = foreground_segmentation(image, segments_full, initial_fg_seeds, initial_bg_seeds)

    print("[graphcut_segmentation] Press any key to close display.")
    show_image_resized("Initial Segmentation", initial_segmented)
    cv2.waitKey(0)
    cv2.destroyWindow("Initial Segmentation")

    all_fg_seeds = set(initial_fg_seeds)
    all_bg_seeds = set(initial_bg_seeds)

    user_cmd = input('[graphcut_segmentation] Refine Segmentation [y/n]? ').lower()
    if user_cmd != 'y':
        print("[graphcut_segmentation] Graph Cut completed!")
        return initial_segmented, initial_mask

    final_segmented, final_mask = initial_segmented, initial_mask
    user_refine = True

    while user_refine:
        final_segmented, final_mask, all_fg_seeds, all_bg_seeds = refine(
            image, image, final_mask, segments_full, all_fg_seeds, all_bg_seeds)

        print("[graphcut_segmentation] Press any key to close display.")
        show_image_resized("Refined Segmentation", final_segmented)
        cv2.waitKey(0)
        cv2.destroyWindow("Refined Segmentation")

        user_cmd = input('[graphcut_segmentation] Continue refining [y/n]? ').lower()
        user_refine = user_cmd == 'y'

    print("[graphcut_segmentation] Graph Cut completed!")
    return final_segmented, final_mask

if __name__ == "__main__":
    import sys
    import os
    # "D:\DATA\CUHKSZ\2024-2025\ECE4513\final-project\cinematic-blur\data\DAVIS\JPEGImages\JPEGImages\480p\blackswan\00000.jpg"
    base_dir = os.path.dirname(os.path.abspath(__file__))  # directory where script is
    image_path = os.path.abspath(os.path.join(base_dir, "../../data/DAVIS/JPEGImages/JPEGImages/480p/blackswan/00000.jpg"))
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image: {image_path}")
        sys.exit(1)
    final_segmented, final_mask = graphcut_segmentation(image)