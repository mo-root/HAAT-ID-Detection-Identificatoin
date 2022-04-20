from flask import Flask, render_template, request
import cv2
import os
import pytesseract
import datetime
from ID_identification import id_identification

# Load pretrained face detection model

UPLOAD_FOLDER = \
    r'C:\Users\Student\PycharmProjects\
    verification-identification\shots'

# These will be later used for downloading the ID images
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__, template_folder='templates',
            static_folder='static')  # Bootstrap(app)

app.config['SECRET_KEY'] = 'super-secret-key'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

pytesseract.pytesseract.tesseract_cmd \
    = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

camera = cv2.VideoCapture(0)


# route to upload the image locally - most efficient
@app.route('/', methods=['GET', 'POST'])
def capturing_cam():
    """
    Main route for the flask API
    You add the image then it is downloaded to a local file
    #note that the local file is uploaded on a cloud meaning that
    downloaded images will be uploaded to the cloud, but are not deleted.
    To delete the images for the sake of privacy and storage save.
    One must add a function to
    Returns:

           """

    if request.method == 'POST':
        photo = request.files['image']
        now = datetime.datetime.now()
        p = os.path.sep.join(
            ['shots', "shot_{}.png".format(str(now).replace(":", ''))])
        print("here is the path to the image", p)
        photo.save(p)
        id_identification(p, 'Biometric_ID')

    return render_template('Pics.html')


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


def upload_file(file):
    if file and allowed_file(file.filename):
        filename = file.filename
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    # note that the local file is uploaded to the used cloud
    # meaning that - downloaded images will be uploaded to the cloud.
    # However, they won't be deleted until the delete function is added.
    # Delete function: os.remove(file_path)


if __name__ == '__main__':
    app.run()

camera.release()
cv2.destroyAllWindows()
