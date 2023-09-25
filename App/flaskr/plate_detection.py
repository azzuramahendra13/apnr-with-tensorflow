import cv2
import numpy as np
import openvino as ov

class PlateDetection:
    def __init__(self, model_xml):
        core = ov.Core()
        devices = core.available_devices

        for device in devices:
            device_name = core.get_property(device, "FULL_DEVICE_NAME")
            print(f"{device}: {device_name}")

        model = core.read_model(model=model_xml)
        
        self.compiled_model = core.compile_model(model=model, device_name="CPU")

    def predict(self, img):
        image_np = cv2.resize(src=img, dsize=(255, 255))
        image_np_expanded = np.expand_dims(image_np, 0)
        
        result = self.compiled_model(image_np_expanded)
        
        return result
    
    def add_detection_box(self, box, image, label = None):
        ymin, xmin, ymax, xmax = box
        pt1, pt2 = (int(xmin), int(ymin)), (int(xmax), int(ymax))

        # box_color = [random.randint(0, 255) for _ in range(3)]
        box_color = [0, 0, 255]
        # line_thickness = round(0.002 * (image.shape[0] + image.shape[1]) / 2) + 1
        line_thickness = 2

        cv2.rectangle(img=image, 
                    pt1=pt1, 
                    pt2=pt2, 
                    color=box_color, 
                    thickness=line_thickness, 
                    lineType=cv2.LINE_AA)

        if label:
            # font_thickness = max(line_thickness - 1, 1)
            font_thickness = 5
            font_face = 0
            font_scale = line_thickness
            font_color = (255, 255, 255)
            text_size = cv2.getTextSize(text=label, 
                                        fontFace=font_face, 
                                        fontScale=font_scale, 
                                        thickness=font_thickness)[0]
            
            rectangle_point1 = pt1
            rectangle_point2 = (pt1[0] + text_size[0], pt1[1] - text_size[1] - 3)
            
            cv2.rectangle(img=image, 
                        pt1=rectangle_point1, 
                        pt2=rectangle_point2, 
                        color=box_color, 
                        thickness=-1,
                        lineType=cv2.LINE_AA)
            
            text_position = pt1[0], pt1[1] - 3

            cv2.putText(img=image, 
                        text=label, 
                        org=text_position, 
                        fontFace=font_face, 
                        fontScale=font_scale, 
                        color=font_color, 
                        thickness=font_thickness, 
                        lineType=cv2.LINE_AA)
            
        return image

    def visualize_inference_result(self, image, labels_map, score, bbox, is_normalized):
        if not bbox.any():
            return image
        
        if is_normalized:
            original_image_height, original_image_width, _ = image.shape
            normalized_detection_boxex = bbox[::] * [
                original_image_height,
                original_image_width,
                original_image_height,
                original_image_width,
            ]
        else:
            normalized_detection_boxex = bbox[::]

        image_with_detection_boxex = np.copy(image)


        detected_class_name = labels_map[1]
        label = f"{detected_class_name} {score:.2f}"
        
        self.add_detection_box(
            box=normalized_detection_boxex[0],
            image=image_with_detection_boxex,
            label=label,
        )
        
        return image_with_detection_boxex

    def old_visualize_inference_result(self, inference_result, image, labels_map, detections_limit = None):
        detection_boxes: np.ndarray = inference_result.get("detection_boxes")
        detection_classes: np.ndarray = inference_result.get("detection_classes")
        detection_scores: np.ndarray = inference_result.get("detection_scores")
        num_detections: np.ndarray = inference_result.get("num_detections")

        detections_limit = int(
            min(detections_limit, num_detections[0])
            if detections_limit is not None
            else num_detections[0]
        )

        # Normalize detection boxes coordinates to original image size
        original_image_height, original_image_width, _ = image.shape
        normalized_detection_boxex = detection_boxes[::] * [
            original_image_height,
            original_image_width,
            original_image_height,
            original_image_width,
        ]

        image_with_detection_boxex = np.copy(image)

        for i in range(detections_limit):
            detected_class_name = labels_map[int(detection_classes[0, i])]
            score = detection_scores[0, i]
            label = f"{detected_class_name} {score:.2f}"

            if score > 0.2:
                self.add_detection_box(
                    box=normalized_detection_boxex[0, i],
                    image=image_with_detection_boxex,
                    label=label,
                )
        
        return image_with_detection_boxex
        # cv2.imshow("Object Detection", image_with_detection_boxex)