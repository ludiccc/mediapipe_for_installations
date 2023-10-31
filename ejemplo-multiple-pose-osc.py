#
#
# This code is a derivative code from the provided in https://developers.google.com/mediapipe/
# The example was made by Federico Joselevich Puiggrós and team 
# within the context of Taller 5, Diseño Multimedial, Facultad de Artes, Universidad Nacional de La Plata
#
#
# This example sends the (x,y) from the nose of each person in front of the camera.
# STEP 1: Import the necessary modules.
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

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
parser.add_argument("--debug_send", action='store_true', help="For debugging everything that is being sent.")
parser.add_argument("--send_separated", action='store_true', help="TODO: Send each person as a different message.")
args = parser.parse_args()

#osc server:
from pythonosc import osc_message_builder
from pythonosc import udp_client
from pythonosc import dispatcher
from pythonosc import osc_server

client = udp_client.SimpleUDPClient(args.ip, args.port)


# landmarks parser:
import random
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np

def draw_landmarks_on_image(rgb_image, detection_result):
    landmarks_list = detection_result.pose_landmarks
    annotated_image = np.copy(rgb_image)

    
    # Loop through the detected poses to visualize.
    print("len(landmarks_list):", len(landmarks_list))

    strmsg = ""
    msg = osc_message_builder.OscMessageBuilder(address = 'poses')
    for idx in range(len(landmarks_list)):
        landmarks = landmarks_list[idx]

        # Draw the pose landmarks.
        landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        landmarks_proto.landmark.extend([
          landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in landmarks
        ])

        solutions.drawing_utils.draw_landmarks(
          annotated_image,
          landmarks_proto,
          solutions.pose.POSE_CONNECTIONS,
          solutions.drawing_styles.get_default_pose_landmarks_style())
        
        if len(strmsg) > 0: strmsg = strmsg + ","
        strmsg = strmsg + "{:.3f}".format(landmarks[0].x)+" "+"{:.3f}".format(landmarks[0].y)+" "+"{:.3f}".format(landmarks[0].z)
        if args.debug_send:
            print("\tLandmark",idx,strmsg)


    msg.add_arg(strmsg, arg_type="s")

    print("Detectadas", len(landmarks_list), "personas")
    print(strmsg)
    msg = msg.build()
    client.send(msg)


    return annotated_image


def detect_and_show(image):
    detection_result = detector.detect(image)

    # STEP 5: Process the detection result. In this case, visualize it.
    annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result)
    if args.image == '': 
        text = str(cap.get(cv2.CAP_PROP_FPS))
        cv2.putText(annotated_image, text, (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Analysis", cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))


VisionRunningMode = mp.tasks.vision.RunningMode

# STEP 2: Create an PoseLandmarker object.
model_path = 'pose_landmarker_lite.task'
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    output_segmentation_masks=True,
    min_pose_detection_confidence=0.5,
    min_pose_presence_confidence=0.5,
    min_tracking_confidence=0.5,
    num_poses=10 # maximum number of poses that can be detected...
    )
detector = vision.PoseLandmarker.create_from_options(options)

# STEP 3: Load the input image.

if args.image != "":
    test_image = mp.Image.create_from_file(args.image)    
    while True:
        # Read frame, crop it, flip it, suits your needs.
        
        detect_and_show(test_image)
        c = cv2.waitKey(1) & 0xFF
        if c == 27 or c == 'q':
            break

else:
    cap = cv2.VideoCapture(args.cam)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.cam_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.cam_height)


    # STEP 4: Detect pose landmarks from the input image.
    while cap.isOpened():
        # Read frame, crop it, flip it, suits your needs.
        res, frame = cap.read()
        frame.flags.writeable = False
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

        detect_and_show(mp_image)

        c = cv2.waitKey(1) & 0xFF
        if c == 27 or c == 'q':
            break

    cap.release()

cv2.destroyAllWindows()


