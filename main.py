import os
import cv2
import numpy as np
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Suppress TensorFlow logging
tf.get_logger().setLevel('ERROR')           # Suppress TensorFlow logging (2)

# Load pipeline config and build a detection model
PATH_TO_CFG = 'workspace/training_demo/exported-models/my_model_mobilenet/pipeline.config'
configs = config_util.get_configs_from_pipeline_file(PATH_TO_CFG)
model_config = configs['model']
detection_model = model_builder.build(model_config=model_config, is_training=False)

# Restore checkpoint
PATH_TO_CKPT = 'workspace/training_demo/exported-models/my_model_mobilenet/checkpoint/'
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(PATH_TO_CKPT, 'ckpt-0')).expect_partial()

@tf.function
def detect_fn(image):
    """Detect objects in image."""

    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)

    return detections, prediction_dict, tf.reshape(shapes, [-1])

PATH_TO_LABELS = 'workspace/training_demo/annotations/label_map.pbtxt'
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

############################

def auto_canny(image, sigma=0.25):
    # compute the median of the single channel pixel intensities
    v = np.median(image)

    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)

    # return the edged image
    return edged

def get_contours_with_auto_canny(orig_img):
    img = orig_img.copy()

    dim1,dim2, _ = img.shape

    # Calculate the width and height of the image
    img_y = len(img)
    img_x = len(img[0])

    #Split out each channel
    blue, green, red = cv2.split(img)
    mn, mx = 220, 350
    # Run canny edge detection on each channel

    blue_edges = auto_canny(blue)

    green_edges = auto_canny(green)

    red_edges = auto_canny(red)

    # Join edges back into image
    edges = blue_edges | green_edges | red_edges

    contours, hierarchy = cv2.findContours(edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    cnts=sorted(contours, key = cv2.contourArea, reverse = True)[:20]
    hulls = [cv2.convexHull(cnt) for cnt in cnts]
    perims = [cv2.arcLength(hull, True) for hull in hulls]
    approxes = [cv2.approxPolyDP(hulls[i], 0.02 * perims[i], True) for i in range(len(hulls))]

    approx_cnts = sorted(approxes, key = cv2.contourArea, reverse = True)
    lengths = [len(cnt) for cnt in approx_cnts]

    approx = approx_cnts[lengths.index(4)]

    return approx

cap = cv2.VideoCapture('App/flaskr/static/img/Video.mp4')

while True:
    # Read frame from camera
    ret, image_np = cap.read()

    if ret:
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)

        # Things to try:
        # Flip horizontally
        # image_np = np.fliplr(image_np).copy()

        # Convert image to grayscale
        # image_np = np.tile(
        #     np.mean(image_np, 2, keepdims=True), (1, 1, 3)).astype(np.uint8)

        input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
        detections, predictions_dict, shapes = detect_fn(input_tensor)

        label_id_offset = 1
        image_np_with_detections = image_np.copy()
        
        viz_utils.visualize_boxes_and_labels_on_image_array(
            image_np_with_detections,
            detections['detection_boxes'][0].numpy(),
            (detections['detection_classes'][0].numpy() + label_id_offset).astype(int),
            detections['detection_scores'][0].numpy(),
            category_index,
            use_normalized_coordinates=True,
            max_boxes_to_draw=200,
            min_score_thresh=.50,
            agnostic_mode=False)

        # Display output
        cv2.imshow('object detection', cv2.resize(image_np_with_detections, (800, 600)))

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