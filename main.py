import cv2
import sys
from mail import sendEmail
import paho.mqtt.client as mqtt
from flask import Flask, render_template, Response
from camera import VideoCamera
import time
import threading


def on_connect(client, userdata, flags, rc):
    print("Connected to MQTT server with result code :"+str(rc))

client = mqtt.Client()
client.on_connect = on_connect
client.connect("192.168.0.xxx", 1883, 60) # use your MQTT server name
client.loop_start()

email_update_interval = 600 # sends an email only once in this time interval
video_camera = VideoCamera(flip=True) # creates a camera object, flip vertically
object_classifier = cv2.CascadeClassifier("models/fullbody_recognition_model.xml") # an opencv classifier

# App Globals (do not edit)
app = Flask(__name__)
last_epoch = 0

def check_for_objects():
	global last_epoch
	while True:
		try:
			frame, found_obj = video_camera.get_object(object_classifier)
			if found_obj and (time.time() - last_epoch) > email_update_interval:
				last_epoch = time.time()
				print("Sending email...")
				sendEmail(frame)
				#cv2.imwrite("/tmp/motion.jpg",frame)
				#tmpFile = open("/tmp/motion.jpg","rb")
				#byteArr=bytearray(tmpFile)
				print("Sending MQTT message...")
				#client.publish("home/door/front/camera",byteArr,0,True)
				client.publish("home/door/front/motion","ON",0,False)
                                client.publish("home/door/front/camera",frame,0,True)

				print("done!")
		except:
			print("Error sending email: ", sys.exc_info()[0])

@app.route('/')
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
