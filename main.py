import cv2
import sys
from mail import sendEmail, sendVideoEmail
from flask import Flask, render_template, Response
from camera import VideoCamera
from flask_basicauth import BasicAuth
import time
import threading
import os

email_update_interval = 600 # sends an email only once in this time interval
video_camera = VideoCamera(flip=False) # creates a camera object, flip vertically
object_classifier = cv2.CascadeClassifier("models/fullbody_recognition_model.xml") # an opencv classifier
use_motion_detection = False

send_video = True
send_video_len = 30 #length of the video attached to the second email
keep_video_after_sending = False

# App Globals (do not edit)
app = Flask(__name__)
app.config['BASIC_AUTH_USERNAME'] = 'CHANGE_ME_USERNAME'
app.config['BASIC_AUTH_PASSWORD'] = 'CHANGE_ME_PLEASE'
app.config['BASIC_AUTH_FORCE'] = True

basic_auth = BasicAuth(app)
last_epoch = 0

def check_for_objects():
    global last_epoch
    while True:
        try:
            global use_motion_detection
            if use_motion_detection:
                frame, found_obj = video_camera.motion_detection()
                if found_obj:
                    # motion detection is fired only if detected in two frames in a row (reduces false positive)
                    frame, found_obj = video_camera.motion_detection()
            else:
                frame, found_obj = video_camera.get_object(object_classifier)
            if found_obj and (time.time() - last_epoch) > email_update_interval:
                last_epoch = time.time()
                print ("Sending email...")
                sendEmail(frame)
                print ("done!")
                if send_video:
                    print ("Capturing video...")
                    vid = video_camera.capture_video(send_video_len)
                    print ("Sending video email...")
                    sendVideoEmail(vid, keep_video_after_sending)
                    print ("done!")
                    if not keep_video_after_sending:
                        os.remove(vid)
                        print ("Video file removed")
        except:
            print ("Error sending email: ", sys.exc_info()[0])

@app.route('/')
@basic_auth.required
def index():
    return render_template('index.html')

def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen(video_camera),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    t = threading.Thread(target=check_for_objects, args=())
    t.daemon = True
    t.start()
    app.run(host='0.0.0.0', debug=False)
