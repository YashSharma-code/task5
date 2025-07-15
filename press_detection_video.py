import cv2
from ultralytics import YOLO

# Load your YOLOv8 model
model = YOLO("best.pt")  # path to your trained weights

# Load input video
cap = cv2.VideoCapture("keyboard_input.mp4")
assert cap.isOpened(), "Error: Could not open video."

# Output video writer (optional)
save_output = True
output_path = "annotated_output.avi"
fps = cap.get(cv2.CAP_PROP_FPS)
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
if save_output:
    out_writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (width, height))

# Class names from your data.yaml
class_names = model.names
pressing_classes =['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'accent', 'ae', 'alt-left', 'altgr-right', 'b', 'c', 'caret', 'comma', 'd', 'del', 'e', 'enter', 'f', 'g', 'h', 'hash', 'i', 'j', 'k', 'keyboard', 'l', 'less', 'm', 'minus', 'n', 'o', 'oe', 'p', 'plus', 'point', 'q', 'r', 's', 'shift-left', 'shift-lock', 'shift-right', 'space', 'ss', 'strg-left', 'strg-right', 't', 'tab', 'u', 'ue', 'v', 'w', 'x', 'y', 'z']


def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = max(1, (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1))
    boxBArea = max(1, (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1))

    return interArea / float(boxAArea + boxBArea - interArea)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=0.4, iou=0.5)
    pressed_keys = []

    for result in results:
        boxes = result.boxes
        key_boxes = []
        press_boxes = []

        for box in boxes:
            cls_id = int(box.cls[0])
            class_name = class_names[cls_id]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            color = (0, 255, 0)

            # Save key or press box
            if class_name in pressing_classes:
                press_boxes.append((class_name, (x1, y1, x2, y2)))
                color = (0, 0, 255)
            else:
                key_boxes.append((class_name, (x1, y1, x2, y2)))

            # Draw bounding box
            label = f"{class_name} {conf:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Detect overlaps between pressing objects and keys
        for press_cls, press_box in press_boxes:
            for key_cls, key_box in key_boxes:
                iou = compute_iou(press_box, key_box)
                if iou > 0.3:
                    pressed_keys.append(key_cls)

    # Display pressed keys
    if pressed_keys:
        display_text = f"Keys Pressed: {', '.join(sorted(set(pressed_keys)))}"
        print(display_text)
        cv2.putText(frame, display_text, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    # Show the frame
    cv2.imshow("Key Press Detection", frame)
    if save_output:
        out_writer.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
if save_output:
    out_writer.release()
cv2.destroyAllWindows()
