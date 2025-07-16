import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation import slic
from skimage.color import label2rgb
from skimage import graph
import maxflow

print("cv2 path:", cv2.__file__)
print("cv2 version:", cv2.__version__)
print(cv2.getBuildInformation())

scribble_mask = []
drawing = False
current_label = 1  # 1 for FG (red), 2 for BG (blue)

def scribble_seeds(event, x, y, flags, param):
    global drawing, current_label

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        current_label = 1  # Foreground
    elif event == cv2.EVENT_RBUTTONDOWN:
        drawing = True
        current_label = 2  # Background
    elif event in [cv2.EVENT_LBUTTONUP, cv2.EVENT_RBUTTONUP]:
        drawing = False
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        cv2.circle(scribble_mask, (x, y), 5, current_label, -1)

def get_user_seeds(image, segments):
    '''
    Get initial seeds/inputted foreground and background information from user.
    '''
    global scribble_mask

    segments = segments.astype(np.int32)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    scribble_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    # Overlay segments for visual context
    seg_vis = label2rgb(segments, image_rgb, kind='avg')
    seg_vis = (seg_vis * 255).astype(np.uint8)
    seg_vis = cv2.cvtColor(seg_vis, cv2.COLOR_RGB2BGR)

    cv2.namedWindow("Scribble on Image")
    cv2.setMouseCallback("Scribble on Image", scribble_seeds)

    while True:
        overlay = seg_vis.copy()
        overlay[scribble_mask == 1] = [0, 0, 255]   # Foreground scribble in red
        overlay[scribble_mask == 2] = [255, 0, 0]   # Background scribble in blue
        cv2.imshow("Scribble on Image", overlay)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break

        cv2.destroyAllWindows()

    # Map scribbled pixels to segment labels
    fg_seeds = set(np.unique(segments[scribble_mask == 1]))
    bg_seeds = set(np.unique(segments[scribble_mask == 2]))

    return fg_seeds, bg_seeds

def mean_shift(image):
    '''
    Initial low level segmentation to speed up graph cut process. Uses SLIC (superpixel clustering)
    from skimage for faster segmentation that the mean shift in OpenCV. 
    '''
    return slic(image, n_segments=500, compactness=10)

def build_RAG(image, segments):
    '''
    Build Region Adjacency Graph
    '''
    rag = graph.rag_mean_color(image, segments, mode='similarity')
    return rag

def iterative_graph_cut(fg_seeds, bg_seeds, rag):
    '''
    Implements the graph cut algorithm with sub-graphs. Minimization of the energy function
    is done using the Boykov-Kolmogorov algorithm to perform min-cut/max-flow on the graph.
    '''
    while True:
        # Build subgraph containing fg/bg seeds + neighbors
        sub_nodes = fg_seeds | bg_seeds
        for n in list(sub_nodes):
            for nb in rag.neighbors(n):
                sub_nodes.add(nb)
        G = rag.subgraph(sub_nodes).copy()

        # COnstruct terminal weights (t-links) & edges between regions (n-links)
        g = maxflow.Graph[float]()
        node_ids = {n:g.add_nodes(1) for n in G.nodes()}
        for n in G.nodes():
            if n in fg_seeds:
                g.add_tedge(node_ids[n], 1e6, 0)
            elif n in bg_seeds:
                g.add_tedge(node_ids[n], 0, 1e6)
            else:
                fg_cost = np.mean(G.nodes[n]['color'][:, 0])
                bg_cost = np.mean(G.nodes[n]['color'][:, 1])
                g.add_tedge(node_ids[n], fg_cost, bg_cost)
        
        for u, v, data in G.edges(data=True):
            w = data['weight']
            g.add_edge(node_ids[u], node_ids[v], w, w)
        
        # Run min-cut / max-flow
        g.maxflow()
        new_fg = {n for n in G.nodes() if g.get_segment(node_ids[n]) == 0}

        # Update seed sets
        new = new_fg - fg_seeds
        if not new:
            break
        fg_seeds |= new
        bg_seeds |= {n for n in G.nodes() if n not in new_fg}

    return fg_seeds, bg_seeds

def segments_to_mask(segments, fg_labels):
    mask = np.zeros_like(segments, dtype=np.uint8)
    for label in fg_labels:
        mask[segments == label] = 255
    return mask

def main():
    img_path = r'D:/DATA/CUHKSZ/2024-2025/ECE4513/final-project/cinematic-blur/images/clay.png'
    image = cv2.imread(img_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    segments = mean_shift(image_rgb)

    fg_seeds, bg_seeds = get_user_seeds(image, segments)
    rag = build_RAG(image_rgb, segments)
    fg_labels = iterative_graph_cut(fg_seeds, bg_seeds, rag)

    mask = segments_to_mask(segments, fg_labels)
    result = cv2.bitwise_and(image, image, mask=mask)

    cv2.imshow("Final Foreground Mask", mask)
    cv2.imshow("Segmented Foreground", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()