# MediaPipe for installations - MediaPipe para instalaciones
Simple MediaPipe examples for electronic/digital/robotic art installations

Ejemplos sencillos para utilizar en instalaciones de arte electrónico/digital/robótico. 

Estos ejemplos son parte del trabajo de la materia Taller 5 de la licenciatura en Diseño Multimedial de la Universidad Nacional de La Plata.


Estos ejemplos son de uso académico. No tomarlos como referencia formal de cómo usar MediaPipe. Para eso, referirse a [https://developers.google.com/mediapipe/](https://developers.google.com/mediapipe/)

Contacto: Federico Joselevich Puiggrós [f@ludic.cc](mailto:f@ludic.cc)
[www.ludic.cc](https://www.ludic.cc)

# How to use this examples - Como usar estos ejemplos (in spanish, just for now...)

Estos ejemplos de scripts de python tienen tres versiones: una para reconocimiento de nodos de cara (face mesh), una para reconocimiento de nodos del cuerpo (pose landmark) y una para reconocimiento de nodos de mano (hand landmarks).

Tomando como ejemplo el reconocimiento de nodos del cuerpo:

Se ejecutan dos procesos, uno en python para hacer el reconocimiento y otro en lo que sea que vaya a recibir los mensajes OSC, por ejemplo, el sketch de Processing que está en este repositorio:

```python ejemplo-pose-landmark-osc.py```

Los mensajes OSC que van desde el script en python contienen una cantidad de argumentos igual a la cantidad de nodos detectados. Es decir, en el caso del *ejemplo-pose-landmark-osc.py*, se reconocen hasta 32 nodos, según ![Pose Landmarks list](https://developers.google.com/static/mediapipe/images/solutions/pose_landmarks_index.png). Entonces, el script mandará un mensaje con 32 argumentos, cada uno conteniendo los tres valores (*x*, *y*, *z*) de posición para ese nodo, separados con espacio. Los valores están normalizados, es decir, valen entre 0 y 1.

En el ejemplo de Processing, estos valores se desglosan y se agregan a un PVector, para después dibujarse en pantalla.