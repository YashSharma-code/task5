import os
import cv2
from ultralytics import YOLO

# Load model
model = YOLO("best.pt")

# Input: image file or directory
input_path = "test_images"  # can be a folder or an image file
output_dir = "annotated_images"
os.makedirs(output_dir, exist_ok=True)

# Press object classes (add your class names here)
pressing_classes =  ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'accent', 'ae', 'alt-left', 'altgr-right', 'b', 'c', 'caret', 'comma', 'd', 'del', 'e', 'enter', 'f', 'g', 'h', 'hash', 'i', 'j', 'k', 'keyboard', 'l', 'less', 'm', 'minus', 'n', 'o', 'oe', 'p', 'plus', 'point', 'q', 'r', 's', 'shift-left', 'shift-lock', 'shift-right', 'space', 'ss', 'strg-left', 'strg-right', 't', 'tab', 'u', 'ue', 'v', 'w', 'x', 'y', 'z']

class_names = model.names

def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = max(1, (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = max(1, (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

    return interArea / float(boxAArea + boxBArea - interArea)

# Get list of images
if os.path.isdir(input_path):
    image_files = [os.path.join(input_path, f) for f in os.listdir(input_path) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
else:
    image_files = [input_path]

# Run inference
for img_path in image_files:
    image = cv2.imread(img_path)
    results = model(image, conf=0.4, iou=0.5)

    for result in results:
        key_boxes = []
        press_boxes = []
        boxes = result.boxes

        for box in boxes:
            cls_id = int(box.cls[0])
            class_name = class_names[cls_id]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])

            # Categorize box
            if class_name in pressing_classes:
                press_boxes.append((class_name, (x1, y1, x2, y2)))
                color = (0, 0, 255)
            else:
                key_boxes.append((class_name, (x1, y1, x2, y2)))
                color = (0, 255, 0)

            # Draw box
            label = f"{class_name} {conf:.2f}"
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(image, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Detect key presses
        pressed_keys = []
        for press_cls, press_box in press_boxes:
            for key_cls, key_box in key_boxes:
                iou = compute_iou(press_box, key_box)
                if iou > 0.3:
                    pressed_keys.append(key_cls)

        if pressed_keys:
            pressed_text = f"Pressed: {', '.join(sorted(set(pressed_keys)))}"
            print(f"{os.path.basename(img_path)} => {pressed_text}")
            cv2.putText(image, pressed_text, (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # Save annotated image
    save_path = os.path.join(output_dir, os.path.basename(img_path))
    cv2.imwrite(save_path, image)
    print(f"Saved: {save_path}")
