import os
import random
import cv2
import numpy as np
import tensorflow as tf

def add_detection_box(box: np.ndarray, image: np.ndarray, label = None) -> np.ndarray:
    """
    Helper function for adding single bounding box to the image

    Parameters
    ----------
    box : np.ndarray
        Bounding box coordinates in format [ymin, xmin, ymax, xmax]
    image : np.ndarray
        The image to which detection box is added
    label : str, optional
        Detection box label string, if not provided will not be added to result image (default is None)

    Returns
    -------
    np.ndarray
        NumPy array including both image and detection box

    """
    ymin, xmin, ymax, xmax = box
    point1, point2 = (int(xmin), int(ymin)), (int(xmax), int(ymax))
    # box_color = [random.randint(0, 255) for _ in range(3)]
    box_color = [150, 150, 150]
    line_thickness = round(0.002 * (image.shape[0] + image.shape[1]) / 2) + 1

    cv2.rectangle(img=image, pt1=point1, pt2=point2, color=box_color, thickness=line_thickness, lineType=cv2.LINE_AA)

    if label:
        font_thickness = max(line_thickness - 1, 1)
        font_face = 0
        font_scale = line_thickness / 3
        font_color = (255, 255, 255)
        text_size = cv2.getTextSize(text=label, fontFace=font_face, fontScale=font_scale, thickness=font_thickness)[0]
        # Calculate rectangle coordinates
        rectangle_point1 = point1
        rectangle_point2 = (point1[0] + text_size[0], point1[1] - text_size[1] - 3)
        # Add filled rectangle
        cv2.rectangle(img=image, pt1=rectangle_point1, pt2=rectangle_point2, color=box_color, thickness=-1, lineType=cv2.LINE_AA)
        # Calculate text position
        text_position = point1[0], point1[1] - 3
        # Add text with label to filled rectangle
        cv2.putText(img=image, text=label, org=text_position, fontFace=font_face, fontScale=font_scale, color=font_color, thickness=font_thickness, lineType=cv2.LINE_AA)
    return image

def visualize_inference_result(inference_result, image: np.ndarray, labels_map, detections_limit = None):
    """
    Helper function for visualizing inference result on the image

    Parameters
    ----------
    inference_result : OVDict
        Result of the compiled model inference on the test image
    image : np.ndarray
        Original image to use for visualization
    labels_map : Dict
        Dictionary with mappings of detection classes numbers and its names
    detections_limit : int, optional
        Number of detections to show on the image, if not provided all detections will be shown (default is None)
    """
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

        if score > 0.6:
            add_detection_box(
                box=normalized_detection_boxex[0, i],
                image=image_with_detection_boxex,
                label=label,
            )
    
    cv2.imshow("Object Detection", image_with_detection_boxex)

# from object_detection.utils import label_map_util
# from object_detection.utils import config_util
# from object_detection.utils import visualization_utils as viz_utils
# from object_detection.builders import model_builder

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Suppress TensorFlow logging
# tf.get_logger().setLevel('ERROR')           # Suppress TensorFlow logging (2)

# # Load pipeline config and build a detection model
# PATH_TO_CFG = 'workspace/training_demo/exported-models/my_model_mobilenet/pipeline.config'
# configs = config_util.get_configs_from_pipeline_file(PATH_TO_CFG)
# model_config = configs['model']
# detection_model = model_builder.build(model_config=model_config, is_training=False)

# # Restore checkpoint
# PATH_TO_CKPT = 'workspace/training_demo/exported-models/my_model_mobilenet/checkpoint/'
# ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
# ckpt.restore(os.path.join(PATH_TO_CKPT, 'ckpt-0')).expect_partial()

# @tf.function
# def detect_fn(image):
#     """Detect objects in image."""

#     image, shapes = detection_model.preprocess(image)
#     prediction_dict = detection_model.predict(image, shapes)
#     detections = detection_model.postprocess(prediction_dict, shapes)

#     return detections, prediction_dict, tf.reshape(shapes, [-1])

# PATH_TO_LABELS = 'workspace/training_demo/annotations/label_map.pbtxt'
# category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

############################

# def auto_canny(image, sigma=0.25):
#     # compute the median of the single channel pixel intensities
#     v = np.median(image)

#     # apply automatic Canny edge detection using the computed median
#     lower = int(max(0, (1.0 - sigma) * v))
#     upper = int(min(255, (1.0 + sigma) * v))
#     edged = cv2.Canny(image, lower, upper)

#     # return the edged image
#     return edged

# def get_contours_with_auto_canny(orig_img):
#     img = orig_img.copy()

#     dim1,dim2, _ = img.shape

#     # Calculate the width and height of the image
#     img_y = len(img)
#     img_x = len(img[0])

#     #Split out each channel
#     blue, green, red = cv2.split(img)
#     mn, mx = 220, 350
#     # Run canny edge detection on each channel

#     blue_edges = auto_canny(blue)

#     green_edges = auto_canny(green)

#     red_edges = auto_canny(red)

#     # Join edges back into image
#     edges = blue_edges | green_edges | red_edges

#     contours, hierarchy = cv2.findContours(edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

#     cnts=sorted(contours, key = cv2.contourArea, reverse = True)[:20]
#     hulls = [cv2.convexHull(cnt) for cnt in cnts]
#     perims = [cv2.arcLength(hull, True) for hull in hulls]
#     approxes = [cv2.approxPolyDP(hulls[i], 0.02 * perims[i], True) for i in range(len(hulls))]

#     approx_cnts = sorted(approxes, key = cv2.contourArea, reverse = True)
#     lengths = [len(cnt) for cnt in approx_cnts]

#     approx = approx_cnts[lengths.index(4)]

#     return approx

# cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

import openvino as ov

core = ov.Core()
devices = core.available_devices

for device in devices:
    device_name = core.get_property(device, "FULL_DEVICE_NAME")
    print(f"{device}: {device_name}")

classification_model_xml = "..\..\openvino_test\model\ir\my_model_mobilenet.xml"

model = core.read_model(model=classification_model_xml)
compiled_model = core.compile_model(model=model, device_name="CPU")

input_layer = model.input(0)
output_layer = model.outputs

N, H, W, C = input_layer.shape

# cap = cv2.VideoCapture('../../openvino_test/Video_resized.mp4')
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while True:
    # Read frame from camera
    ret, frame = cap.read()

    if ret:
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        
        image_np = cv2.resize(src=frame, dsize=(255, 255))
        image_np_expanded = np.expand_dims(image_np, 0)
        
        result = compiled_model(image_np_expanded)
        image_detection_boxes = result[1]
        image_detection_classes = result[2]
        image_detection_scores = result[4]
        image_num_detections = result[5]


        # Things to try:
        # Flip horizontally
        # image_np = np.fliplr(image_np).copy()

        # Convert image to grayscale
        # image_np = np.tile(
        #     np.mean(image_np, 2, keepdims=True), (1, 1, 3)).astype(np.uint8)

        # input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
        # detections, predictions_dict, shapes = detect_fn(input_tensor)

        # label_id_offset = 1
        image_np_with_detections = image_np.copy()
        
        # viz_utils.visualize_boxes_and_labels_on_image_array(
        #     image_np_with_detections,
        #     detections['detection_boxes'][0].numpy(),
        #     (detections['detection_classes'][0].numpy() + label_id_offset).astype(int),
        #     detections['detection_scores'][0].numpy(),
        #     category_index,
        #     use_normalized_coordinates=True,
        #     max_boxes_to_draw=200,
        #     min_score_thresh=.50,
        #     agnostic_mode=False)

        # Display output
        labels_map = {1: 'plat-nomor'}

        visualize_inference_result(
            inference_result=result,
            image=cv2.resize(frame, (640, 480),interpolation= cv2.INTER_LINEAR),
            labels_map=labels_map,
            detections_limit=5,
)

        # plat_nomor_label = 0
        # detections_plat_nomor_idx = np.where(detections['detection_classes'][0] == 0)[0]        
        
        # detections_plat_nomor = {'boxes': tf.zeros((1, len(detections_plat_nomor_idx))), 
        #                         'classes': tf.zeros((1, len(detections_plat_nomor_idx))),
        #                         'scores': tf.zeros((1, len(detections_plat_nomor_idx)))
        #                         }
        
        # for idx in range(len(detections_plat_nomor_idx)):
        #     detections_plat_nomor['boxes'][0][idx].assign(detections['detection_boxes'][0][detections_plat_nomor_idx[idx]])
        #     detections_plat_nomor['classes'][0][idx].assign(detections['detection_classes'][0][detections_plat_nomor_idx[idx]])
        #     detections_plat_nomor['scores'][0][idx].assign(detections['detection_scores'][0][detections_plat_nomor_idx[idx]])

        # print(detections_plat_nomor['boxes'])
        # Display cropped output
        #check the ratio of the detected plate area to the bounding box
        # image_np_copy = image_np.copy()

        # approx = get_contours_with_auto_canny(image_np)
        
        # if (cv2.contourArea(approx)/(image_np.shape[0]*image_np.shape[1]) > .2):
        #     cv2.drawContours(image_np_copy, [approx], -1, (0,255,0), 1)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()