import cv2
import mediapipe as mp
from argparse import ArgumentParser
from pythonosc import udp_client, osc_message_builder

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

with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
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
            for landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)

                msg = osc_message_builder.OscMessageBuilder(address='hand_landmarks')

                for landmark in landmarks.landmark:
                    strmsg = "{:.3f}".format(landmark.x)+" "+"{:.3f}".format(landmark.y)+" "+"{:.3f}".format(landmark.z)
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
