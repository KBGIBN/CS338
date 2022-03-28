import cv2
import numpy as np

video = cv2.VideoCapture("D:/UIT/CS338/line_detector/road_video.mp4")
frame_width = int(video.get(3))
frame_height = int(video.get(4))
out = cv2.VideoWriter('D:/UIT/CS338/line_detector/result.mp4',cv2.VideoWriter_fourcc('m','p','4','v'), 10, (frame_width,frame_height))

while True:
    ret, orig_frame = video.read()
    if not ret:
        video = cv2.VideoCapture("D:/UIT/CS338/line_detector/road_video.mp4")
        continue

    frame = cv2.GaussianBlur(orig_frame, (5, 5), 0)
    
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    low_yellow = np.array([20, 100, 100])
    up_yellow = np.array([30, 255, 255])
    mask = cv2.inRange(hsv, low_yellow, up_yellow)

    edges = cv2.Canny(mask, 75, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, maxLineGap=50)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 5)

    out.write(frame)

    cv2.imshow("frame", frame)
    key = cv2.waitKey(15)
    if key == 27:
        break

    

video.release()
out.release()
cv2.destroyAllWindows()