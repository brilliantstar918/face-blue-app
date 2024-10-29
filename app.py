import os
import cv2
import dlib
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
import imutils

# Initialize Flask app
app = Flask(__name__)

# Configure upload folders
UPLOAD_FOLDER = 'static/uploads/'
PROCESSED_FOLDER = 'static/results/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'mp4', 'avi'}

# Helper function to check file extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Route for home page
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle file uploads
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Check if the file is an image or video
        if filename.lower().endswith(('png', 'jpg', 'jpeg')):
            output_filename = blur_faces_in_image(filepath)
        else:
            output_filename = blur_faces_in_video(filepath)

        return redirect(url_for('processed_file', filename=output_filename))

    return redirect(request.url)

# Process the image and blur faces
def blur_faces_in_image(image_path):
    detector = dlib.get_frontal_face_detector()
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray, 1)

    for face in faces:
        x, y, w, h = (face.left(), face.top(), face.width(), face.height())
        face_region = image[y:y+h, x:x+w]
        blurred_face = cv2.GaussianBlur(face_region, (25, 25), 0)
        image[y:y+h, x:x+w] = blurred_face

    # Save the processed image
    output_filename = "blurred_" + os.path.basename(image_path)
    output_path = os.path.join(app.config['PROCESSED_FOLDER'], output_filename)
    cv2.imwrite(output_path, image)
    
    return output_filename

# Process the video and blur faces in each frame
def blur_faces_in_video(video_path):
    detector = dlib.get_frontal_face_detector()
    video_capture = cv2.VideoCapture(video_path)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_filename = "blurred_" + os.path.basename(video_path)
    output_path = os.path.join(app.config['PROCESSED_FOLDER'], output_filename)
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray, 1)

        for face in faces:
            x, y, w, h = (face.left(), face.top(), face.width(), face.height())
            face_region = frame[y:y+h, x:x+w]
            blurred_face = cv2.GaussianBlur(face_region, (25, 25), 0)
            frame[y:y+h, x:x+w] = blurred_face
        
        out.write(frame)

    video_capture.release()
    out.release()

    print('Finished!')

    return output_filename

# Serve the processed file
@app.route('/processed/<filename>')
def processed_file(filename):
    return send_from_directory(app.config['PROCESSED_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=False)
