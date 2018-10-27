import cv2
from imutils.video.pivideostream import PiVideoStream
import imutils
import time
import numpy as np
import datetime

video_res = (640, 480)
object_det_res = (320, 240)

video_frame_rate = 30
video_out_dir = '/home/pi/'

background_frame = None
background_update_rate = 120 # 2 mins. Period to update background frame for motion detection.
last_back_update = None
motion_det_min_area = 2000 # Regulate motion detection sensitivity. Smaller value - greater sensitivity.

class VideoCamera(object):
    def __init__(self, flip = False):
        self.vs = PiVideoStream(resolution=video_res, framerate=video_frame_rate).start()
        self.flip = flip
        
        self.x_coef = float(video_res[0]) / object_det_res[0] # needed to translate between frame sizes
        self.y_coef = float(video_res[1]) / object_det_res[1]
        
        time.sleep(2.0)

    def __del__(self):
        self.vs.stop()

    def flip_if_needed(self, frame):
        if self.flip:
            return np.flip(frame, 0)
        return frame

    def get_frame(self):
        frame = self.flip_if_needed(self.vs.read())
        ret, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()

    def motion_detection(self):
        frame = self.flip_if_needed(self.vs.read()) #grabbing frame
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #graying it
        gray = cv2.GaussianBlur(gray, (5, 5), 0) #blurring
        
        global background_frame
        global last_back_update
        global background_update_rate
        global motion_det_min_area
        
        if (background_frame is None) or (time.time() - last_back_update) > background_update_rate:
            background_frame = gray
            last_back_update = time.time()
            return None, False
            
        frameDelta = cv2.absdiff(background_frame, gray) 
        thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]
        
        thresh = cv2.dilate(thresh, None, iterations=2) #fill the holes
        (_, cnts, _) = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #find contours of the objects
        
        found_something = False
        
        return_obj = frame
        # return_obj = gray
        
        for c in cnts:
            # if the contour is too small, ignore it
            if cv2.contourArea(c) < motion_det_min_area:
                continue
                
            found_something = True
            (x, y, w, h) = cv2.boundingRect(c)
            cv2.rectangle(return_obj, (x, y), (x + w, y + h), (153, 0, 204), 2) # different color rectangle for motion detect
            
        ret, jpeg = cv2.imencode('.jpg', return_obj)
        return (jpeg.tobytes(), found_something)

    def get_object(self, classifier):
        found_objects = False
        frame = self.flip_if_needed(self.vs.read()).copy() 
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, object_det_res, interpolation=cv2.INTER_AREA)

        objects = classifier.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        if len(objects) > 0:
            found_objects = True

        # Draw a rectangle around the objects
        for (x, y, w, h) in objects:
            cv2.rectangle(frame
                , (int(self.x_coef * x), int(self.y_coef * y))
                , (int(self.x_coef * x + self.x_coef * w), int(self.y_coef * y + self.y_coef * h))
                , (0, 255, 0)
                , 2)

        ret, jpeg = cv2.imencode('.jpg', frame)
        return (jpeg.tobytes(), found_objects)
        
    def capture_video(self, send_video_len):
        video_loc = video_out_dir + datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S") +'.avi'
        fourcc = cv2.VideoWriter_fourcc(*'X264')
        out = cv2.VideoWriter(video_loc, fourcc, video_frame_rate, video_res)
        
        start_time = time.time()
        while (time.time() - start_time) <= send_video_len:
            frame = self.flip_if_needed(self.vs.read())
            #(_, _, frame) = self.motion_detection()
            out.write(frame)
        out.release()
        
        return video_loc


