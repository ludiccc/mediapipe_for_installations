#
#
#
# This code is a derivative code from the provided in https://developers.google.com/mediapipe/
# The example was made by Federico Joselevich Puiggrós and team 
# within the context of Taller 5, Diseño Multimedial, Facultad de Artes, Universidad Nacional de La Plata
#
#
# This example sends the (x,y) each person face in front of the camera.
# STEP 1: Import the necessary modules.
import cv2
from typing import Tuple, Union
import math
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2

#arguments:
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--cam", type=int, default=0,
                    help="The webcam index.")
parser.add_argument('--cam_width', type=int, default=640)
parser.add_argument('--cam_height', type=int, default=480)
parser.add_argument("--ip", default="127.0.0.1", help="The ip of the OSC server")
parser.add_argument("--port", type=int, default=5005, help="The port the OSC server is listening on")
parser.add_argument("--image", type=str, default="", help="For testing porpouses only, just test an image.")
args = parser.parse_args()

#osc server:
from pythonosc import osc_message_builder
from pythonosc import udp_client
from pythonosc import dispatcher
from pythonosc import osc_server

client = udp_client.SimpleUDPClient(args.ip, args.port)

TEXT_COLOR = (255, 0, 0)
MARGIN = 10  # pixels
ROW_SIZE = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# STEP 2
BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

base_options = python.BaseOptions(model_asset_path='gesture_recognizer.task')
options = vision.GestureRecognizerOptions(base_options=base_options)
recognizer = vision.GestureRecognizer.create_from_options(options)


# STEP 3: Load the input image.
cap = cv2.VideoCapture(args.cam)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.cam_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.cam_height)

if args.image != "":
    test_image = mp.Image.create_from_file(args.image)    

# STEP 4: Detect pose landmarks from the input image.
while cap.isOpened():
    # Read frame, crop it, flip it, suits your needs.
    res, frame = cap.read()
    frame.flags.writeable = False
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)


    with GestureRecognizer.create_from_options(options) as recognizer:    
        # STEP 4: Recognize gestures in the input image.
        recognition_result = recognizer.recognize(mp_image)


        if len(recognition_result.gestures) > 0:

            
            msg = osc_message_builder.OscMessageBuilder(address = 'gestures')

            strmsg = f"{recognition_result.gestures[0][0].category_name} ({recognition_result.gestures[0][0].score:.2f})"    
            print(strmsg)
            msg.add_arg(strmsg, arg_type="s")

            cv2.putText(frame, strmsg, (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)


            for hand_landmarks in recognition_result.hand_landmarks:
                    hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
                    hand_landmarks_proto.landmark.extend([
                        landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
                    ])

                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks_proto,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())

                    for landmark in hand_landmarks:
                        strmsg = "{:.3f}".format(landmark.x)+" "+"{:.3f}".format(landmark.y)+" "+"{:.3f}".format(landmark.z)
                        print(strmsg)
                        msg.add_arg(strmsg, arg_type="s")


            msg = msg.build()
            client.send(msg)


    # Display the frame
    cv2.imshow('Hand Landmarks', frame)

    c = cv2.waitKey(1) & 0xFF
    if c == 27 or c == 'q':
        break



cv2.destroyAllWindows()
cap.release()

