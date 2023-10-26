# MediaPipe for installations - MediaPipe para instalaciones
Con el propósito principal de este repositorio es dar ejemplos para el trabajo de estudiantes que hablan castellano, la documentación principal está en ese idioma. Se pueden referir a ella aquí: [Castellano]: https://github.com/ludiccc/mediapipe_for_installations/blob/main/README.md

Simple MediaPipe examples for electronic/digital/robotic art installations

These examples are part of the work of the subject Workshop 5/Final Project Workshop of the degree in Multimedia Design at the National University of La Plata.

These examples are for academic use. Do not take them as a formal reference on how to use MediaPipe. For that, refer to [https://developers.google.com/mediapipe/](https://developers.google.com/mediapipe/)


Contacto/Contact: Federico Joselevich Puiggrós [f@ludic.cc](mailto:f@ludic.cc)
[www.ludic.cc](https://www.ludic.cc)

# How to use this examples

These python script examples have three versions: one for face landmarks recognition (face mesh), one for body landmarks recognition (pose landmark), and one for hand landmarks recognition (hand landmarks).

Taking the recognition of body nodes as an example:

Two processes are executed, one in python to do the recognition and another in whatever is going to receive the OSC messages, for example, the Processing sketch that is in this repository:

```python ejemplo-pose-landmark-osc.py```

The OSC messages coming from the python script contain a number of arguments equal to the number of detected nodes. That is, in the case of *ejemplo-pose-landmark-osc.py*, up to 32 nodes are recognized, according to ![Pose Landmarks list](https://developers.google.com/static/mediapipe/images/solutions/pose_landmarks_index.png). The script will then send a message with 32 arguments, each containing the three position values (*x*, *y*, *z*) for that node, separated by a space. The values are normalized, that is, they are between 0 and 1.

In the Processing example, these values are broken down and added to a PVector, then drawn on the screen.