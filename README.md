# 🎯 YOLO Keyboard Key Detection (Part 1)

This project uses YOLO to detect individual physical keyboard keys from a top-down camera view.

✅ Completed: **Part 1 – Key Detection**  

---

## 📂 Files in This Repo

| File                         | Description                                      |
|------------------------------|--------------------------------------------------|
| `best.pt`                    | Trained YOLO model weights                       |
| `press_detection_image.py`   | Script to test key detection on images           |
| `press_detection_video.py`   | Script to test key detection on video files      |

---
An example output image is given in the repo as example.jpg
---
## 🛠️ Requirements

Install required packages:

```bash
pip install ultralytics opencv-python
```
🧪 How to Run
🔹 Run on Images
Detect keys and save annotated output images:

```
python press_detection_image.py
```
Edit the image_path variable in the script to point to your image file or place your images in a "test_images" folder in the same directory as the launch file.

Your output images will be stored in a folder named annotated_images in the working directory.
🔹 Run on Video
Detect keys in a recorded video:

```
python press_detection_video.py 
```
Edit the video_path variable in the script to point to your .mp4 or .avi video file or place your video renamed as "keyboard_input" in the same directory as the launch file.
Your output video will be stored as annotated_output.avi in the working directory.
