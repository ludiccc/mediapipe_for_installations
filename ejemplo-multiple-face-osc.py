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


def _normalized_to_pixel_coordinates(
    normalized_x: float, normalized_y: float, image_width: int,
    image_height: int) -> Union[None, Tuple[int, int]]:
  """Converts normalized value pair to pixel coordinates."""

  # Checks if the float value is between 0 and 1.
  def is_valid_normalized_value(value: float) -> bool:
    return (value > 0 or math.isclose(0, value)) and (value < 1 or
                                                      math.isclose(1, value))

  if not (is_valid_normalized_value(normalized_x) and
          is_valid_normalized_value(normalized_y)):
    # TODO: Draw coordinates even if it's outside of the image bounds.
    return None
  x_px = min(math.floor(normalized_x * image_width), image_width - 1)
  y_px = min(math.floor(normalized_y * image_height), image_height - 1)
  return x_px, y_px

def visualize(
    image,
    detection_result
) -> np.ndarray:
  """Draws bounding boxes and keypoints on the input image and return it.
  Args:
    image: The input RGB image.
    detection_result: The list of all "Detection" entities to be visualize.
  Returns:
    Image with bounding boxes.
  """
  annotated_image = image.copy()
  height, width, _ = image.shape

  # Loop through the detected poses to visualize.
  print("len(detection_result.detections):", len(detection_result.detections))

  strmsg = ""
  msg = osc_message_builder.OscMessageBuilder(address = 'faces')

  for detection in detection_result.detections:
    # Draw bounding_box
    bbox = detection.bounding_box
    start_point = bbox.origin_x, bbox.origin_y
    end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
    cv2.rectangle(annotated_image, start_point, end_point, TEXT_COLOR, 3)

    # Draw keypoints
    for keypoint in detection.keypoints:
      keypoint_px = _normalized_to_pixel_coordinates(keypoint.x, keypoint.y,
                                                     width, height)
      color, thickness, radius = (0, 255, 0), 2, 2
      cv2.circle(annotated_image, keypoint_px, thickness, color, radius)

    # Draw label and score
    category = detection.categories[0]
    category_name = category.category_name
    category_name = '' if category_name is None else category_name
    probability = round(category.score, 2)
    result_text = category_name + ' (' + str(probability) + ')'
    text_location = (MARGIN + bbox.origin_x,
                     MARGIN + ROW_SIZE + bbox.origin_y)
    cv2.putText(annotated_image, result_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                FONT_SIZE, TEXT_COLOR, FONT_THICKNESS)

    if len(strmsg) > 0: strmsg = strmsg + ","
    strmsg = strmsg + "{:.3f}".format(keypoint.x)+" "+"{:.3f}".format(keypoint.y)

  print(strmsg)
  msg = msg.build()
  client.send(msg)

  return annotated_image


# STEP 2: Create an FaceDetector object.
BaseOptions = mp.tasks.BaseOptions
FaceDetector = mp.tasks.vision.FaceDetector
FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Create a face detector instance with the image mode:
options = FaceDetectorOptions(
    base_options=BaseOptions(model_asset_path='blaze_face_short_range.tflite'),
    running_mode=VisionRunningMode.IMAGE)
detector = vision.FaceDetector.create_from_options(options)

# STEP 3: Load the input image.
cap = cv2.VideoCapture(args.cam)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.cam_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.cam_height)

def detect_and_show(image):
    detection_result = detector.detect(image)

    # STEP 5: Process the detection result. In this case, visualize it.
    annotated_image = visualize(image.numpy_view(), detection_result)
    text = "Anotated:" + str(cap.get(cv2.CAP_PROP_FPS))
    cv2.putText(annotated_image, text, (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)


    #cv2.imshow("Analysis", cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
    #cv2.imshow("test", cv2.cvtColor(annotated_image_test, cv2.COLOR_RGB2BGR))
    #outputImg = np.hstack((cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR), cv2.cvtColor(annotated_image_test, cv2.COLOR_RGB2BGR)))
    #cv2.imshow("Ejemplo Multiple Face", outputImg)
    cv2.imshow("Ejemplo Multiple Face", cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))


if args.image != "":
    test_image = mp.Image.create_from_file(args.image)    

    while True:
        # Read frame, crop it, flip it, suits your needs.
        
        detect_and_show(test_image)
        c = cv2.waitKey(1) & 0xFF
        if c == 27 or c == 'q':
            break

else:
    # STEP 4: Detect pose landmarks from the input image.
    while cap.isOpened():
        # Read frame, crop it, flip it, suits your needs.
        res, frame = cap.read()
        frame.flags.writeable = False
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)


        detect_and_show(frame)
        c = cv2.waitKey(1) & 0xFF
        if c == 27 or c == 'q':
            break


cv2.destroyAllWindows()
cap.release()

