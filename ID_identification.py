# Import packages
from read_from_image import extract_numbers
from Image_classification import card_classification
from utils import label_map_util, visualization_utils as vis_util
import tensorflow as tf
import cv2
import os
import numpy as np

# The model is injected as soon as the program starts running
# then it doesn't do the activity more,
# unless you reRun the program.

# Loading the model of the AI (open-source).
CWD_PATH = os.getcwd()

# Name of the model of the open-source AI (ID detection)
MODEL_NAME = 'model'
PATH_TO_CKPT = os.path.join(CWD_PATH, MODEL_NAME, 'frozen_inference_graph.pb')
# Path to label map file.
PATH_TO_LABELS = os.path.join(CWD_PATH, 'data', 'labelmap.pbtxt')
# Number of classes the object detector can identify.
# Meaning how many objects to identify
NUM_CLASSES = 1
# Load the label map.
# Label maps map indices to category names, so that when our convolution
# network predicts `5`, we know that this corresponds to `king`.
# Here we use internal utility functions, but anything that returns a
# dictionary mapping integers to appropriate string labels would be fine.
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(
    label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)
# Load the Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.compat.v1.GraphDef()
    with tf.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.compat.v1.Session(graph=detection_graph)

image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
num_detections = detection_graph.get_tensor_by_name('num_detections:0')


def id_identification(path_to_id):
    """
    This function checks, according to
    an open sourced and trained AI
    if the given photo contains an ID card
    and measures the percentage of this being an ID card
    Args:
        path_to_id(str): String containing the path file to the image

    Returns: if the image contains an ID card it takes us to the
    Image classifier.
    If not, it asks the user to take another photo.

    """
    # Name of the directory containing the object detection module we're using
    IMAGE_NAME = path_to_id
    # Grab path to current working directory
    # Path to frozen detection graph .pb file, which contains the model
    # for object detection.
    # Path to image
    PATH_TO_IMAGE = os.path.join(CWD_PATH, IMAGE_NAME)
    # Define input and output tensors (i.e. data) for the object detection
    # classifier
    # Input tensor is the image
    image = cv2.imread(PATH_TO_IMAGE)

    image_expanded = np.expand_dims(image, axis=0)
    # Perform the actual detection by running the model with the image as input
    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: image_expanded})
    # print(image_tensor)
    # ##print('a very important note regarding the np.squeeze')
    # #print(scores)
    # #print(detection_scores)
    # #print("These are the scores", num_detections)
    # Draw the results of the detection (aka 'visualize the results')
    _, array_coord = vis_util.visualize_boxes_and_labels_on_image_array(
        image,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=80,
        min_score_thresh=0.999999)

    percentage = ' '.join([str(elem) for elem in scores[0]])
    percent = (extract_numbers(percentage))
    # percent is the probability this is an Israeli ID
    listToStr = ''.join([str(elem) for elem in percent])
    # print("this is the list", listToStr)
    percent = float("." + listToStr)
    # print("this is an Israeli ID or Driving license: ", percent * 100, "%.")
    # the scores are given by the array
    if percent > 0.099999:
        # if the identification is more than 5 9s
        # this is the ideal number for making sure
        card_classification(path_to_id)
    else:
        # print("the image does not contain a real ID")
        return 'please take another pic'
