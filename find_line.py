import cv2
cap = cv2.VideoCapture("data/demo_tracking.mp4")
ret, frame = cap.read()

def click_event(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Point cliqué : ({x}, {y})")
        cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)
        cv2.imshow("Image", frame)

def main():
    cv2.imshow("Image", frame)
    cv2.setMouseCallback("Image", click_event)
    cv2.waitKey(0)
    
if __name__ == "__main__":
    main()

