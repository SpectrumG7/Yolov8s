from ultralytics import YOLO
import cv2
import numpy as np

# Load YOLO model
model = YOLO(r'C:\Users\752595\best\best.pt')

# Open Vedipo Camera for real time detection 
cap = cv2.VideoCapture(0)

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

    # Algorithm for crack detections
    for box, score, cls_id in zip(boxes, scores, classes):
        if score < CONFIDENCE_THRESHOLD:
            continue
 
        defect_detected = True
        x1, y1, x2, y2 = map(int, box)
 
        label = f"{class_names[int(cls_id)]} {score:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 1)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Convert to be HSV for definde golden region of PZT
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_gold = np.array([15, 50, 100])
    upper_gold = np.array([35, 255, 255])
    gold_mask = cv2.inRange(hsv, lower_gold, upper_gold)

    # Detect rectangle gray stript (by optimize range grayscale)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    lower_gray = np.array([100])
    upper_gray = np.array([150])
    gray_mask = cv2.inRange(gray, lower_gray, upper_gray)

    # Integrating golden mask and cut-off gray stript
    combined_mask = cv2.bitwise_and(gold_mask, cv2.bitwise_not(gray_mask))

    # adjust contrast for emphersize crack region
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)

    # Detect crack line by using Canny
    edges = cv2.Canny(enhanced, 50, 150)

    # Using mask for reemphersize PZT crack black line over golden region
    defect = cv2.bitwise_and(edges, edges, mask=combined_mask)

    # Identification the contour of verifcle  rectangle 
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 5000:  # filter out the small region
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = w / float(h)
            if 0.28 < aspect_ratio < 0.95 and h > w:  # Detect rectangle in vertical plan 
                # Covers a rectangular area
                roi = defect[y:y+h, x:x+w]
                roi_contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for rc in roi_contours:
                    if cv2.contourArea(rc) > 100:  # Filter out small crack area
                        rc += [x, y]  # Adjust contour coordinates
                        #cv2.drawContours(frame, [rc], -1, (0, 0, 255), 2)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Display the result 
    cv2.imshow('u-PZT Crack Defect Detection', frame)

    # Exit the program by pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Close camera and window
cap.release()
cv2.destroyAllWindows()
