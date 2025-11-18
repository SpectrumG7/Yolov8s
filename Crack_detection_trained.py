import cv2
import os
import json
import numpy as np
from ultralytics import YOLO

# ───── Configuration ─────
model = YOLO(r'C:\Users\752595\best\weights_V3.pt')
save_dir = r'C:\Users\752595\best_failed_capture'
json_output_dir = 'json_results'
os.makedirs(save_dir, exist_ok=True)
os.makedirs(json_output_dir, exist_ok=True)
CONFIDENCE_THRESHOLD = 0.50
SHARPNESS_THRESHOLD = 1000

# ───── Video Capture ─────
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

out = cv2.VideoWriter(
    r'C:\Users\752595\video_failed_capture\output_video.mp4v',
    cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height)
)

frame_count = 0
save_count = 0
prev_detection_state = None

# ───── Utility Functions ─────
def calculate_sharpness(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    return laplacian.var()

def draw_overlay(frame, text, position, color=(255, 255, 255)):
    cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

# ───── Main Loop ─────
while True:
    start_time = cv2.getTickCount()
    ret, frame = cap.read()
    if not ret:
        break

    # ───── YOLO Detection ─────
    results = model.predict(source=frame, imgsz=640, conf=0.20, save=False, verbose=False)[0]
    boxes = results.boxes.xyxy.cpu().numpy()
    scores = results.boxes.conf.cpu().numpy()
    classes = results.boxes.cls.cpu().numpy()
    class_names = results.names

    defect_detected = False
    for box, score, cls_id in zip(boxes, scores, classes):
        if score < CONFIDENCE_THRESHOLD:
            continue
        defect_detected = True
        x1, y1, x2, y2 = map(int, box)
        label = f"{class_names[int(cls_id)]} {score:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 1)
        draw_overlay(frame, label, (x1, y1 - 10), (0, 0, 255))

    # ───── Confidence & FPS ─────
    avg_confidence = round(float(np.mean(scores)) * 100, 2) if len(scores) > 0 else 0.0
    end_time = cv2.getTickCount()
    time_elapsed = (end_time - start_time) / cv2.getTickFrequency()
    fps_value = round(1.0 / time_elapsed, 2)

    # ───── Result Text ─────
    if avg_confidence < 50:
        result_text = 'Resolution Adjusting'
        text_color = (255, 0, 0)
        save_label = "LowConfidence"
    elif defect_detected:
        result_text = 'Acceptance Result: Reject'
        text_color = (0, 0, 255)
        save_label = "Rejected"
    else:
        result_text = 'Acceptance Result: Accept'
        text_color = (0, 255, 0)
        save_label = "Accept"

    draw_overlay(frame, result_text, (10, height - 40), text_color)
    draw_overlay(frame, f'FPS: {fps_value}', (10, height - 70), (0, 255, 255))
    draw_overlay(frame, f'Confidence: {avg_confidence}%', (10, height - 100), (255, 255, 0))

    # ───── Save Frame & JSON ─────
    current_detection_state = {
        'defect_detected': defect_detected,
        'total_detections': len(boxes)
    }

    if current_detection_state != prev_detection_state:
        save_path = os.path.join(save_dir, f'{save_label}_{frame_count:05d}.jpg')
        cv2.imwrite(save_path, frame)
        save_count += 1

        result_data = {
            "frame_id": frame_count,
            "acceptance_result": save_label,
            "scores": round(float(np.mean(scores)), 2)
        }
        json_path = os.path.join(json_output_dir, f"result_{frame_count:05d}.json")
        with open(json_path, 'w') as f:
            json.dump(result_data, f, indent=4)

        prev_detection_state = current_detection_state

    cv2.imshow('u-PZT Crack Defect Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    frame_count += 1

# ───── Cleanup ─────
cap.release()
cv2.destroyAllWindows()
print(f"\nTotal frames saved: {save_count}")
