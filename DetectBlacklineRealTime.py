import cv2
import numpy as np
from yolox.exp import get_exp
from yolox.utils import get_model

def detect_black_lines_realtime_yolox():
  #Load model YOLOX 
  exp = get_exp(None, "yolox-s")
  model = get_model(exp)
  model.eval()
  
  # Load weights and chang the directory 
  ckpt = "yolox_s.pth"
  model.load_weights(ckpt)
  
  # Open Camera
  cap = cv2.VideoCapture(0)
  if not cap.isOpened():
    print("Unable to opened the Camera")
    return
    
  while True:
    ret, frame = cap.read()
    if not ret:
      print("Unable to read the frame")
      break
      
    # Covert stuckture to be HSV and build mask
    hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    lower_gold = np.array([20, 100, 100])
    upper_gold = np.array([30, 255, 255})
    gold_mask = cv2.inRange(hsv, lower_gold, upper_gold)
    black_mask = cv2.bitwise_not(gold_mask)
    black_mask = medianBlur(black_mask, 5)
    edges = cv2.Canny(black_mask, 50, 150)
    contours, _ = cv2.findContours(edges,
                  cv2.RETR_EXTERNAL,
                  cv2.CHAIN_APPROX_SIMPLE)
    
    # Drawing the ellipse
    for contour in contours:
      if cv2.contourArea(contour) < 100:
        continue
      try:
        ellipse = cv2.fitEllipse(contour)
        cv2.ellipse(frame, ellipse, (0, 255, 0). 2)
      except:
        continue
        
    # Display image 
    cv2.imshow('Real-time Detection', frame)
    cv2.imshow('Black Lines Mask',black_mask)

    if cv2.waitKey(1) & 0xFF == ord('q'):
      break
      
  cap.release()
  cv2.destroyAllWindow()
if __name__ == "__main__":
  
detect_black_lines_reatime_yolox()
      
      
    
    
    
# test  
  

