# MediaPipe for installations - MediaPipe para instalaciones
As the main propouse of this examples is to guide spanish-speaken students, the main documentation will be in spanish. If you need it in english, you have it here: 
[English](https://github.com/ludiccc/mediapipe_for_installations/blob/main/README_en.md)

 _English_
Simple MediaPipe examples for electronic/digital/robotic art installations

These examples are part of the work of the subject Workshop 5/Final Project Workshop of the degree in Multimedia Design at the Universidad Nacional de La Plata: [Taller 5](https://taller5.ludic.cc)

These examples are for academic use. Do not take them as a formal reference on how to use MediaPipe. For that, refer to [https://developers.google.com/mediapipe/](https://developers.google.com/mediapipe/)


_Castellano_
Ejemplos sencillos para utilizar en instalaciones de arte electrónico/digital/robótico. 

Estos ejemplos son parte del trabajo de la materia Taller 5 de la licenciatura en Diseño Multimedial de la Universidad Nacional de La Plata. [Taller 5](https://taller5.ludic.cc)


Estos ejemplos son de uso académico. No tomarlos como referencia formal de cómo usar MediaPipe. Para eso, referirse a [https://developers.google.com/mediapipe/](https://developers.google.com/mediapipe/)

Contacto/Contact: xxx xxx xxx [user@example.com](mailto:user@example.com)
[www.ludic.cc](https://www.ludic.cc)

# mediapipe_osc.py: la app todo-en-uno

`mediapipe_osc.py` reúne los ejemplos de abajo en una sola app configurable:
pose + manos + gestos + cara en un mismo proceso, argumentos OSC numéricos,
IDs estables por persona/mano/cara con eventos `enter`/`leave` y velocidad,
un panel de control en la misma ventana de la vista previa (detectores,
fuente de video — lista de cámaras y Kinects disponibles: varias a la vez se
analizan lado a lado, en el orden que se elija —, pausa, IP/puerto, formatos,
espejo — todo modificable en vivo), entrada de Kinect (v2 y v1) con la opción
**Z from Kinect** (la z de cada punto pasa a ser la distancia medida en
metros), modo espejo, suavizado, modo sin ventana, y formatos compatibles con
los sketches de Processing existentes (`--format legacy`) y con los scripts de
`Sonido_Cuadrafonico` (`--format kinect`, una webcam reemplaza a la Kinect).

Además trae el detector **Blobs** (OpenCV, sin modelos): para una cámara en
el techo, siluetas u objetos detecta las formas que se destacan del fondo,
las enumera y las sigue con IDs (`/mp/blob/...`, y `/kinect/person` para los
scripts de sonido cuadrafónico, como hacía la Kinect original). Y **plantillas**
para cambiar el aspecto de la ventana (`--template minimal | ascii | baroque`,
o un archivo propio; ver `templates/`).

También puede correr sobre un video o una imagen en vez de la webcam, y tiene
tests. El manual en castellano está en
[README_mediapipe_osc_es.md](README_mediapipe_osc_es.md) (su versión en inglés:
[README_mediapipe_osc.md](README_mediapipe_osc.md)). La explicación de cómo
funciona por dentro, pensada para estudiantes, está en
[GUIDE_mediapipe_osc_es.md](GUIDE_mediapipe_osc_es.md) (English:
[GUIDE_mediapipe_osc.md](GUIDE_mediapipe_osc.md)).

```
python mediapipe_osc.py --pose --hands --mirror
python mediapipe_osc.py --list-cameras                      # qué cámaras hay
python mediapipe_osc.py --blobs --format kinect --port 12000   # cámara cenital -> Sonido_Cuadrafonico
python mediapipe_osc.py --template ascii                    # otro aspecto
python osc_monitor.py          # en otra terminal: muestra lo que llega
```

# Como usar estos ejemplos

Para comenzar, es indispensable tener instalado Python y un entorno con Mediapipe. Pueden seguir la guía oficial de instalación de Mediapipe [https://developers.google.com/mediapipe/solutions/setup_python](https://developers.google.com/mediapipe/solutions/setup_python)

Estos ejemplos de scripts de python tienen tres versiones: una para reconocimiento de nodos de cara (face mesh), una para reconocimiento de nodos del cuerpo (pose landmark) y una para reconocimiento de nodos de mano (hand landmarks).

Tomando como ejemplo el reconocimiento de nodos del cuerpo:

Se ejecutan dos procesos, uno en python para hacer el reconocimiento y otro en lo que sea que vaya a recibir los mensajes OSC, por ejemplo, el sketch de Processing que está en este repositorio:

```python ejemplo-pose-landmark-osc.py```

Los mensajes OSC que van desde el script en python contienen una cantidad de argumentos igual a la cantidad de nodos detectados. Es decir, en el caso del *ejemplo-pose-landmark-osc.py*, se reconocen hasta 32 nodos, según ![Pose Landmarks list](https://developers.google.com/static/mediapipe/images/solutions/pose_landmarks_index.png). Entonces, el script mandará un mensaje con 32 argumentos, cada uno conteniendo los tres valores (*x*, *y*, *z*) de posición para ese nodo, separados con espacio. Los valores están normalizados, es decir, valen entre 0 y 1.

En el ejemplo de Processing, estos valores se desglosan y se agregan a un PVector, para después dibujarse en pantalla.
