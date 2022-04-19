from flask import Flask, render_template, request
from Programs.ID_identification import *
import pytesseract
import datetime

# Load pretrained face detection model

UPLOAD_FOLDER = \
    r'C:\Users\Student\PycharmProjects\
    verification-identification\shots'

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
    if request.method == 'POST':
        photo = request.files['image']
        entered_Id = request.form['rid']
        now = datetime.now()
        p = os.path.sep.join(
            ['shots', "shot_{}.png".format(str(now).replace(":", ''))])
        print("here is the path to the image", p)
        photo.save(p)
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
