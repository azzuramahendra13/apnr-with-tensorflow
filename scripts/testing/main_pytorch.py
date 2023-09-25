import cv2
import time
import numpy as np
import sys
from openvino.preprocess import PrePostProcessor
from openvino.runtime import Core, Layout, PartialShape, Type
# from ultralytics import YOLO



# 'App/flaskr/static/img/Video.mp4'
# cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
# cap.set(3, 640)
# cap.set(4, 480)

# frame_width = int(cap.get(3))
# frame_height = int(cap.get(4))

# while cap.isOpened():
#     ret, frame = cap.read()

#     if ret:
#         image = frame
#         image_height, image_width, _ = image.shape


#         start = time.time()
#         results = model(image, stream=True)
#         end = time.time()

#         fps = 1 / (end-start)
        
#         for result in results:
#             boxes = result.boxes
            
#             for box in results.boxes:
#                 class_name = results.names[box.cls[0].item()]
#                 cords = box.xyxy[0].tolist()
#                 cords = [round(x) for x in cords]
#                 conf = round(box.conf[0].item(), 2)
#                 color = COLORS[int(box.cls[0].item())]

#                 x1 = cords[0]
#                 y1 = cords[1]
#                 w = cords[2] - cords[0]
#                 h = cords[3] - cords[1]

#                 cv2.rectangle(image, (x1, y1), (w, h), color, thickness=2)
#                 cv2.putText(image, class_name, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
#                 cv2.putText(image, f"{fps:.2f} FPS", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

#         cv2.imshow('image', image)

#         if cv2.waitKey(10) & 0xFF == ord('q'):
#             break    
#     else:
#         break

# cap.release()
# cv2.destroyAllWindows()

