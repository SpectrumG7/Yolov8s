from ultralytics import YOLO
import cv2
import os
 
# Load YOLO model
model = YOLO(r'C:\Users\752595\best\best.pt')
 
# Open video file
video_path = r'C:\Users\505310\Desktop\Code\Pzt_raw\Video\pzt_video.MOV'
save_dir = r'C:\Users\752595\best_failed_capture'
os.makedirs(save_dir, exist_ok=True)
 
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()
 
# Get video properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
 
# Video writer
out = cv2.VideoWriter(
    r'C:\Users\505310\Desktop\Code\Pzt_Crack\Video\output_video.mov',
    cv2.VideoWriter_fourcc(*'mp4v'),
    fps,
    (width, height)
)
 
# Define vertical zones based on percentage of width
zone1 = (int(0.15 * width), 0, int(0.40 * width), height)
zone2 = (int(0.55 * width), 0, int(0.80 * width), height)
 
# Function to check if a detection box overlaps with a zone
def is_in_zone(box, zone):
    x1, y1, x2, y2 = box
    zx1, zy1, zx2, zy2 = zone
    inter_x1 = max(x1, zx1)
    inter_y1 = max(y1, zy1)
    inter_x2 = min(x2, zx2)
    inter_y2 = min(y2, zy2)
    inter_width = max(0, inter_x2 - inter_x1)
    inter_height = max(0, inter_y2 - inter_y1)
    inter_area = inter_width * inter_height
    return inter_area > 0
 
# Frame and save counters
frame_count = 0
save_count = 0
CONFIDENCE_THRESHOLD = 0.5
 
# Track previous detection state
prev_detection_state = None
 
while True:
    ret, frame = cap.read()
    if not ret:
        break
 
    # Run YOLO prediction
    results = model.predict(source=frame, imgsz=640, conf=0.25, save=False, verbose=False)
    result = results[0]
    boxes = result.boxes.xyxy.cpu().numpy()
    scores = result.boxes.conf.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy()
    class_names = result.names
 
    defect_detected = False
    detection_count = len(boxes)
 
    # Draw vertical zones
    cv2.rectangle(frame, (zone1[0], zone1[1]), (zone1[2], zone1[3]), (0, 255, 0), 2)
    cv2.putText(frame, 'Zone 1', (zone1[0] + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.rectangle(frame, (zone2[0], zone2[1]), (zone2[2], zone2[3]), (0, 255, 0), 2)
    cv2.putText(frame, 'Zone 2', (zone2[0] + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
 
    # Zone detection counts
    zone1_count, zone2_count = 0, 0
    zone1_flag, zone2_flag = False, False
 
    # Process detections
    for box, score, cls_id in zip(boxes, scores, classes):
        if score < CONFIDENCE_THRESHOLD:
            continue
 
        defect_detected = True
        x1, y1, x2, y2 = map(int, box)
 
        if is_in_zone([x1, y1, x2, y2], zone1):
            zone1_count += 1
            zone1_flag = True
        elif is_in_zone([x1, y1, x2, y2], zone2):
            zone2_count += 1
            zone2_flag = True
 
        label = f"{class_names[int(cls_id)]} {score:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 1)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
 
    if zone1_flag:
        cv2.rectangle(frame, (zone1[0], zone1[1]), (zone1[2], zone1[3]), (0, 0, 255), 6)
        cv2.putText(frame, f'Zone 1: {zone1_count} defect', (zone1[0] + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    if zone2_flag:
        cv2.rectangle(frame, (zone2[0], zone2[1]), (zone2[2], zone2[3]), (0, 0, 255), 6)
        cv2.putText(frame, f'Zone 2: {zone2_count} defect', (zone2[0] + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
 
    # Build current detection state summary
    current_detection_state = {
        'defect_detected': defect_detected,
        'zone1_count': zone1_count,
        'zone2_count': zone2_count,
        'total_detections': detection_count
    }
 
    # Save frame if new detection state differs from last saved one
    if defect_detected and (current_detection_state != prev_detection_state):
        save_path = os.path.join(save_dir, f'frame_{frame_count:05d}.jpg')
        cv2.imwrite(save_path, frame)
        save_count += 1
        prev_detection_state = current_detection_state
 
    # Show detection counts
    cv2.putText(frame, f'Total Detections: {detection_count}', (10, height - 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
 
 
 
    # Display and write frame
    # Display and write frame
    height, width = frame.shape[:2]
    half_width = width // 1
    half_height = height // 1
 
    # Display and write frame
    # Resize the window to half size
    cv2.namedWindow('Automate u-PZT/m-PZT crack, ABS defect detections', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Automate u-PZT/m-PZT crack, ABS defect detections', half_width, half_height)
 
    # Show the frame
    cv2.imshow('Automate u-PZT/m-PZT crack, ABS defect detections', frame)
 
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
 
    frame_count += 1
 
# Clean up
cap.release()
out.release()
cv2.destroyAllWindows()
 
print(f"\nTotal frames saved: {save_count}")