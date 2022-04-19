from datetime import datetime
import os
import pytesseract
from Image_classification import *
from read_from_image import extract_numbers
from read_from_image import read_from_image
import cv2

# the model
net = cv2.dnn.readNetFromCaffe(
    './saved_model/ deploy.prototxt.txt',
    './saved_model/res10_300x300_ssd_iter_140000.caffemodel')


def id_numbers_locate(path_to_image, coordination):
    """
        Cuts the image in a way that only the ID numbers would
        be the only text visible in the image

    Args:
        path_to_image(str): path to ID image
        coordination(str): set of variables to
        find and local the ID

    Returns:
        return a cropped image with a frame of the face only
        with the ID under it
    """
    original_image = cv2.imread(path_to_image)
    (h, w) = original_image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(original_image, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    # print(frame.shape, 'this is the shape of the frame')
    # print(h, ' h ', w ,'w')
    detections = net.forward()
    confidence = detections[0, 0, 0, 2]

    if confidence < 0.5:
        return original_image

    box = detections[0, 0, 0, 3:7] * np.array([w, h, w, h])
    (startX, startY, endX, endY) = box.astype("int")

    original_image = original_image[
                     startY: int(endY * 1.5),
                     int(0.9 * startX):int(endX * 1.1)]
    (h, w) = original_image.size
    r = 480 / float(h)
    dim = (int(w * r), 480)
    original_image = cv2.resize(original_image, dim)

    print(len(extract_numbers(pytesseract.image_to_string(original_image))),
          'the length of the read ID')
    cv2.imshow('cropped Image', original_image)
    cv2.waitKey(1000)
    cv2.destroyAllWindows()
    if len(extract_numbers(pytesseract.image_to_string(original_image))) >= 9:
        print('image is valid')
        now = datetime.now()
        path_cropped_image = os.path.sep.join(
            ['shots', "shot_{}-cropped.png".format(str(now).replace(":", ''))])
        print('reading image...')
        cv2.imwrite(path_cropped_image, original_image)
        cv2.imshow('cropped Image', original_image)
        cv2.waitKey(1000)
        cv2.destroyAllWindows()

        read_from_image(path_cropped_image, path_to_image)
    else:
        print('flipping the image')
        flipped_image = cv2.imread(path_to_image)
        flipped_image = cv2.rotate(flipped_image, cv2.ROTATE_180)
        (h, w) = flipped_image.shape[:2]

        blob = cv2.dnn.blobFromImage(cv2.resize(flipped_image, (300, 300)), 1.0,
                                     (300, 300), (104.0, 177.0, 123.0))
        net.setInput(blob)
        # print(frame.shape, 'this is the shape of the frame')
        # print(frame.shape, 'this is the shape of the frame')
        # print(h, ' h ', w ,'w')
        detections = net.forward()
        confidence = detections[0, 0, 0, 2]
        if confidence < 0.5:
            return flipped_image

        box = detections[0, 0, 0, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")
        # the original coordination of the image
        # and we multiply it with the coordination that help us
        # get only the ID
        flipped_image = flipped_image[
                        int(startY * coordination[0]):
                        int(endY * coordination[1]),
                        int(coordination[2] * startX):
                        int(endX * coordination[3])]
        (h, w) = flipped_image.size
        r = 480 / float(h)
        dim = (int(w * r), 480)
        flipped_image = cv2.resize(flipped_image, dim)
        # except Exception as e:
        #     pass

        cv2.imshow('flipped and cropped image', flipped_image)
        cv2.waitKey(1000)
        cv2.destroyAllWindows()
        if len(extract_numbers(pytesseract.image_to_string(flipped_image))) >= 9:
            now = datetime.now()
            path_cropped_image = os.path.sep.join(
                ['shots', "shot_{}-cropped.png".
                    format(str(now).replace(":", ''))])
            cv2.imwrite(path_cropped_image, flipped_image)
            cv2.imshow('flipped and cropped image', flipped_image)
            cv2.waitKey(1000)
            cv2.destroyAllWindows()
            read_from_image(path_cropped_image, path_to_image)
        else:
            print('OCR could\'t find an ID, please try another Image')
            return 'cannot find an ID, please try another Image'
