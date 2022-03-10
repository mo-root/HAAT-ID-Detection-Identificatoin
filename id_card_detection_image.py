# Import packages
from utils import label_map_util, visualization_utils as vis_util
import time
from datetime import datetime
from threading import Thread
import tensorflow as tf
import cv2
import os
import sys
import numpy as np
import io
import requests
import pytesseract
from PIL import Image
from flask import Flask, render_template, Response, request

sys.path.append("")


# The "split" function returns only the numbers in an array
# out of all the charts found in an image.
def extract_numbers(word):
    arr = [char for char in word]
    print(arr)
    arr_clean = list()
    for elm in arr:
        try:
            float(elm)
            print("could convert string to float:", elm)
            arr_clean.append(elm)
        except ValueError as e:
            pass
    ID = ''.join(str(x) for x in arr_clean)
    print(arr_clean, ID)
    return ID


# The model is injected as soon as the program starts running
# then it doesn't do the activity more,
# unless you reRun the program.

# Loading the model of the AI (open-source).
CWD_PATH = os.getcwd()

# Name of the model from GitHub.
MODEL_NAME = 'model'
PATH_TO_CKPT = os.path.join(CWD_PATH, MODEL_NAME, 'frozen_inference_graph.pb')
# Path to label map file.
PATH_TO_LABELS = os.path.join(CWD_PATH, 'data', 'labelmap.pbtxt')
# Number of classes the object detector can identify.
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


# making sure the given pic contains a real israeli ID/Driving license
def id_identification(path_to_id):
    # Name of the directory containing the object detection module we're using
    IMAGE_NAME = path_to_id
    # Grab path to current working directory
    # Path to frozen detection graph .pb file, which contains the model that is used
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
    print(image_tensor)
    # print('a very important note regarding the np.squeeze')
    # print(scores)
    # print(detection_scores)
    # print("These are the scores", num_detections)
    # Draw the results of the detection (aka 'visulaize the results')
    image, array_coord = vis_util.visualize_boxes_and_labels_on_image_array(
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
    listToStr = ''.join([str(elem) for elem in percent])
    print("this is the list", listToStr)
    percent = float("." + listToStr)
    print("this is an Israeli ID or Driving license: ", percent, "%.")
    # the scores are given by the array
    if percent > 0.099999:
        # if the identification is more than 5 9s
        id_numbers_locate(path_to_id)
    else:
        print("the image does not contain a real ID")
        return 'please take another pic'


def validate_id(ID):
    if (len(ID) != 9):
        return False
    IdList = list()
    try:
        id = list(map(int, ID))
    except BaseException:
        return False

    counter = 0

    for i in range(9):
        id[i] *= (i % 2) + 1
        if (id[i] > 9):
            id[i] -= 9
        counter += id[i]

    if (counter % 10 == 0):
        return True
    else:
        return False


def id_numbers_locate(path_to_image):
    global net
    image = cv2.imread(path_to_image)
    (h, w) = image.shape[:2]

    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    # print(frame.shape, 'this is the shape of the frame')
    # print(h, ' h ', w ,'w')
    detections = net.forward()
    confidence = detections[0, 0, 0, 2]

    if confidence < 0.5:
        return image

    box = detections[0, 0, 0, 3:7] * np.array([w, h, w, h])
    (startX, startY, endX, endY) = box.astype("int")
    try:
        image = image[startY: int(endY * 1.5),
                      int(0.8 * startX):int(endX * 1.1)]
        (h, w) = image.size
        r = 480 / float(h)
        dim = (int(w * r), 480)
        image = cv2.resize(image, dim)
    except Exception as e:
        pass
    now = datetime.now()
    path_cropeed_image = os.path.sep.join(
        ['shots', "shot_{}-cropped.png".format(str(now).replace(":", ''))])
    cv2.imwrite(path_cropeed_image, image)
    read_from_image(path_cropeed_image, path_to_image)


def cut_to_face(path):
    face_cascade = cv2.CascadeClassifier('face_detector.xml')
    img = cv2.imread(path)

    faces = face_cascade.detectMultiScale(img, 1.1, 4)

    # Draw rectangle around the faces
    for (x, y, w, h) in faces:
        # cv2.rectangle(img, (x - 10, y), (x + w + 20, y + h * 2), (255, 0, 0), 2)
        pass
    cropped = img[y: y + h, x: x + w]
    cv2.imshow('image', cropped)
    now = datetime.now()
    path_cropeed = os.path.sep.join(
        ['shots', "shot_{}-cropped.png".format(str(now).replace(":", ''))])
    cv2.imwrite(path_cropeed, cropped)
    read_from_image(path_cropeed, path)


def read_from_image(p, og):
    img = Image.open(p)
    # enhancer = ImageEnhance.Contrast(img)
    Actual_ID = 0
    i = 0
    # img = img.crop((width / 1.666, 0, width, height))

    while i < 4:
        # # img = enhancer.enhance(1.5)
        img.show()
        text = pytesseract.image_to_string(img)
        # print('THis s the text with the link', text)
        fID = extract_numbers(text)
        x = len(fID) - 9
        y = len(fID)
        while (x > 0):
            print(fID[x:y],
                  "this is what is being checked at the moment.........")
            if validate_id(fID[x:y]):
                print(fID, 'has the id', fID[x:y])
                return 'has the ID'
                break
            x -= 1
            y -= 1
        img = img.transpose(Image.ROTATE_90)
        i += 1
    print('take another pic please')
    # if the program doesn't find a face

    img = Image.open(og)
    # enhancer = ImageEnhance.Contrast(img)
    Actual_ID = 0
    i = 0
    # img = img.crop((width / 1.666, 0, width, height))

    while i < 4:
        # # img = enhancer.enhance(1.5)
        img.show()
        text = pytesseract.image_to_string(img)
        # print('THis s the text with the link', text)
        fID = extract_numbers(text)
        x = len(fID) - 9
        y = len(fID)
        while (x > 0):
            print(fID[x:y],
                  "this is what is being checked at the moment.........")
            if validate_id(fID[x:y]):
                print(fID,
                      'has the id::: please check the admin dashboard',
                      fID[x:y])
                return 'has the ID'
                break
            x -= 1
            y -= 1
        img = img.transpose(Image.ROTATE_90)
        i += 1
    print('take another pic please')


# for the demo
global capture, rec_frame, grey, switch, neg, face, rec, out
capture = 0
grey = 0
neg = 0
face = 1
switch = 1
rec = 0

# make shots directory to save pics
try:
    os.mkdir('shots')
except OSError as error:
    pass

# Load pretrained face detection model
net = cv2.dnn.readNetFromCaffe(
    './saved_model/deploy.prototxt.txt',
    './saved_model/res10_300x300_ssd_iter_140000.caffemodel')

UPLOAD_FOLDER = r'C:\Users\Student\PycharmProjects\verification-identification\shots'

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

app = Flask(__name__, template_folder='templates',
            static_folder='static')  # Bootstrap(app)

app.config['SECRET_KEY'] = 'super-secret-key'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

camera = cv2.VideoCapture(0)


def record(out):
    global rec_frame
    while rec:
        time.sleep(0.05)
        out.write(rec_frame)


def detect_face(frame):
    global net
    (h, w) = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    # print(frame.shape, 'this is the shape of the frame')
    # print(h, ' h ', w ,'w')
    detections = net.forward()
    confidence = detections[0, 0, 0, 2]

    if confidence < 0.5:
        return frame

    box = detections[0, 0, 0, 3:7] * np.array([w, h, w, h])
    (startX, startY, endX, endY) = box.astype("int")
    try:
        frame = frame[startY + 50:endY + 150, startX:endX]
        (h, w) = frame.size
        r = 480 / float(h)
        dim = (int(w * r), 480)
        frame = cv2.resize(frame, dim)
    except Exception as e:
        pass
    return frame


def gen_frames():
    # generate frame by frame from camera
    global out, capture, rec_frame
    while True:
        success, frame = camera.read()
        frame = cv2.flip(frame, 1)
        if success:
            if face:
                frame = detect_face(frame)

            if grey:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if neg:
                frame = cv2.bitwise_not(frame)
            if capture:
                capture = 0
                now = datetime.now()
                p = os.path.sep.join(
                    ['shots', "shot_{}.png".format(str(now).replace(":", ''))])
                print("heereee is the f path", p)
                cv2.imwrite(p, cv2.flip(frame, 1))
                result = id_identification(p)
                print("result is", result)
                return result
            if (rec):
                rec_frame = frame
                frame = cv2.putText(
                    cv2.flip(
                        frame,
                        1),
                    "Recording...",
                    (0,
                     25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0,
                     0,
                     255),
                    4)
                frame = cv2.flip(frame, 1)

            try:
                ret, buffer = cv2.imencode('.jpg', cv2.flip(frame, 1))
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            except Exception as e:
                pass

        else:
            pass


# Main route
@app.route('/')
def index():
    return render_template('index.html')


# returns the video (Camera capture - live)
@app.route('/video_feed')
def video_feed():
    return Response(
        gen_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame')


# takes a snapshot on the capture click button
@app.route('/requests', methods=['POST', 'GET'])
def tasks():
    global switch, camera
    if request.method == 'POST':
        if request.form.get('click') == 'Capture':
            global capture
            capture = 1
        elif request.form.get('grey') == 'Grey':
            global grey
            grey = not grey
        elif request.form.get('neg') == 'Negative':
            global neg
            neg = not neg
        elif request.form.get('face') == 'Face Only':
            global face
            face = not face
            if (face):
                time.sleep(4)
        elif request.form.get('stop') == 'Stop/Start':

            if (switch == 1):
                switch = 0
                camera.release()
                cv2.destroyAllWindows()

            else:
                camera = cv2.VideoCapture(0)
                switch = 1
        elif request.form.get('rec') == 'Start/Stop Recording':
            global rec, out
            rec = not rec
            if (rec):
                now = datetime.datetime.now()
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                out = cv2.VideoWriter(
                    'vid_{}.avi'.format(
                        str(now).replace(
                            ":", '')), fourcc, 20.0, (640, 480))
                # Start new thread for recording the video
                thread = Thread(target=record, args=[out, ])
                thread.start()
            elif (rec == False):
                out.release()

    elif request.method == 'GET':
        return render_template('index.html')
    return render_template('index.html')


# all routes contain the same runtime because the image is downloaded and
# then the detection runs through the image.


# route to send the image as a link
@app.route('/idl', methods=['GET', 'POST'])
def link_capture():
    if request.method == 'POST':
        imgL = request.form['image']
        response = requests.get(imgL)
        photo = Image.open(io.BytesIO(response.content))
        now = datetime.now()
        p = os.path.sep.join(
            ['shots', "shot_{}.png".format(str(now).replace(":", ''))])
        print("here is the path to the image", p)
        photo.save(p)
        id_identification(p)

    return render_template('Links.html')


# route to upload the iamge locally - most effecient
@app.route('/idc', methods=['GET', 'POST'])
def capturing_cam():
    if request.method == 'POST':
        photo = request.files['image']
        entered_Id = request.form['rid']
        now = datetime.now()
        p = os.path.sep.join(
            ['shots', "shot_{}.png".format(str(now).replace(":", ''))])
        print("here is the path to the image", p)
        photo.save(p)
        if entered_Id == '':
            entered_Id = '214549297'
        id_identification(p)

    return render_template('Pics.html')


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


def upload_file(file):
    if file and allowed_file(file.filename):
        filename = file.filename
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))


if __name__ == '__main__':
    app.run()

camera.release()
cv2.destroyAllWindows()
