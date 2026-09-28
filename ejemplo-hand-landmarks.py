#
#
# This code is a derivative code from the provided in https://developers.google.com/mediapipe/
# The example was made by Federico Joselevich Puiggrós and team 
# within the context of Taller 5, Diseño Multimedial, Facultad de Artes, Universidad Nacional de La Plata
#
#
import cv2
import mediapipe as mp
from argparse import ArgumentParser
from pythonosc import udp_client, osc_message_builder
import math

# Argument parser
parser = ArgumentParser()
parser.add_argument("--cam", type=int, default=0, help="The webcam index.")
parser.add_argument('--cam_width', type=int, default=640)
parser.add_argument('--cam_height', type=int, default=480)
parser.add_argument("--ip", default="127.0.0.1", help="The ip of the OSC server")
parser.add_argument("--port", type=int, default=5005, help="The port the OSC server is listening on")
args = parser.parse_args()

# Initialize OSC client
client = udp_client.SimpleUDPClient(args.ip, args.port)

# Initialize mediapipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Initialize video capture
cap = cv2.VideoCapture(args.cam)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.cam_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.cam_height)

with mp_hands.Hands(
    max_num_hands=6,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the BGR image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the image with mediapipe
        results = hands.process(rgb_frame)

        # Draw the hand landmarks
        if results.multi_hand_landmarks:
            for mano_idx, landmarks in enumerate(results.multi_hand_landmarks):
                mp_drawing.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)

                msg = osc_message_builder.OscMessageBuilder(address='hand_landmarks')

                for idx, landmark in enumerate(landmarks.landmark):
                    strmsg = "{mano_idx} {idx} {landmark.x:.3f} {landmark.y:.3f} {landmark.z:.3f}".format(mano_idx=mano_idx, idx=idx, landmark=landmark)
                    print(strmsg)
                    msg.add_arg(strmsg, arg_type="s")                    

                msg = msg.build()
                client.send(msg)

                msg = osc_message_builder.OscMessageBuilder(address='distancia_pulgar_indice')

                world_landmarks = results.multi_hand_world_landmarks[mano_idx]
                x1 = world_landmarks.landmark[4].x
                y1 = world_landmarks.landmark[4].y
                # z1 = world_landmarks.landmark[4].z
                x2 = world_landmarks.landmark[8].x
                y2 = world_landmarks.landmark[8].y
                # z2 = world_landmarks.landmark[8].z
                distnacia = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                print("Distancia entre 4 y 8:", distnacia)
                print("Mano:", mano_idx)
                strmsg = "{mano_idx} {distancia}".format(mano_idx=mano_idx, distancia=distnacia)
                print(strmsg)
                msg.add_arg(strmsg, arg_type="s")
                msg = msg.build()
                client.send(msg)


        # Display the frame
        cv2.imshow('Hand Landmarks', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


# Release the video capture and close the OpenCV windows
cap.release()
cv2.destroyAllWindows()
