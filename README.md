# cinematic-blur
 Background blur plays a major role in making videos cinematic, but most phone cameras cannot achieve it due to limited depth of field. Editing it manually takes time, and deep learning tools like Rembg and Mediapipe add complexity and require large models.  Therefore, in this work, we propose a simpler alternative using classical image processing and computer vision techniques.

##
## Results Overview
Here are a few of the blurred background results compared to the original and highlighted foreground videos. More of these can be accessed in the outputs folder.

https://github.com/user-attachments/assets/aac4fedc-a7b5-4337-9956-3ac579d940b3

https://github.com/user-attachments/assets/3972b4d4-c970-40c2-add3-91b0c8dfb616

https://github.com/user-attachments/assets/f6115b5f-1b20-479b-926e-1e6be784afcc

https://github.com/user-attachments/assets/ac4b2e9c-226b-41c1-abce-77a6e3ae46e3

# 
## Project Pipeline
<img width="1920" height="1080" alt="project_pipeline" src="https://github.com/user-attachments/assets/4440d336-70ff-4747-a190-403c2f87412c" />

#
## Usage Guide

1. Clone the repository

   ```bash
   git clone https://github.com/kaylinnvk/cinematic-blur 

2. (Optional) Create a new virtual environment
   ```bash
   python -m venv venv-cinematic-blur

   # activate venv (Windows)
   venv-cinematic-blur\Scripts\activate

   # or on macOS/Linux
   source venv-cinematic-blur/bin/activate
   ```

3. Install all requirements
   ```bash
   pip install -r requirements.txt
   ```
   
5. Run ```codes/run.py```.
   You will be prompted to scribble to define the initial foreground and background seeds.
   The program will then ask if you want to refine the segmentation.
   Afterward, it proceeds to mask propagation and background blurring.

   You can modify the specific video and output paths in run.py:
   ```bash
   # modify these lines
   video_path = os.path.abspath(os.path.join(base_dir, "path/to/your/video"))
   output_path = os.path.abspath(os.path.join(base_dir, "path/to/desired/output/path"))
   ```
