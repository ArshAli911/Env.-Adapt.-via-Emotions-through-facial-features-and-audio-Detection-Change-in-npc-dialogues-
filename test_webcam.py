import cv2
 
cap = cv2.VideoCapture(0)
print("Opened:", cap.isOpened())
ret, frame = cap.read()
print("Frame shape:", frame.shape if ret else "No frame")
cap.release() 