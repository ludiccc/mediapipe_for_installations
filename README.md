# MediaPipe para instalaciones

 _English_
Simple MediaPipe examples for electronic/digital/robotic art installations

These examples are for academic use. Do not take them as a formal reference on how to use MediaPipe. For that, refer to [https://developers.google.com/mediapipe/](https://developers.google.com/mediapipe/)


_Castellano_
Ejemplos sencillos para utilizar en instalaciones de arte electrónico/digital/robótico. 

Estos ejemplos son de uso académico. No tomarlos como referencia formal de cómo usar MediaPipe. Para eso, referirse a [https://developers.google.com/mediapipe/](https://developers.google.com/mediapipe/)

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
tests. El manual (en inglés) está en
[README_mediapipe_osc.md](README_mediapipe_osc.md) y la explicación de cómo
funciona por dentro, pensada para estudiantes, en
[GUIDE_mediapipe_osc.md](GUIDE_mediapipe_osc.md).

```
python mediapipe_osc.py --pose --hands --mirror
python mediapipe_osc.py --list-cameras                      # qué cámaras hay
python mediapipe_osc.py --blobs --format kinect --port 12000   # cámara cenital -> Sonido_Cuadrafonico
python mediapipe_osc.py --template ascii                    # otro aspecto
python osc_monitor.py          # en otra terminal: muestra lo que llega
```

# Ejemplos de receptores: Python, Processing, openFrameworks, TouchDesigner

Para cada detector de `mediapipe_osc.py` — **cuerpo** (pose), **manos**,
**gestos de mano**, **cara** y **blobs** — hay un ejemplo receptor en cada uno
de los cuatro entornos que se usan en la materia.  Son el punto de partida
para una instalación: reciben los mensajes OSC, mantienen a cada
persona / mano / cara / blob por su id y dibujan lo que llega; cada uno
imprime además los *eventos* típicos (alguien entró, una pinza se cerró, un
gesto cambió, una boca se abrió, un blob cambió de zona).

| carpeta | qué hay | cómo se prueba |
|---------|---------|----------------|
| [`python-examples/`](python-examples/) | 5 scripts (python-osc + OpenCV) | `make all` los corre uno tras otro; `make test` sin cámara |
| [`processing-examples/`](processing-examples/) | 5 sketches (oscP5) | el IDE de Processing, o `make all` con `processing-java` |
| [`openFrameworks-examples/`](openFrameworks-examples/) | 5 proyectos (ofxOsc), con Makefile + generador de proyectos Xcode / Visual Studio | `make all` dentro de `<openFrameworks>/apps/` |
| [`touchdesigner-examples/`](touchdesigner-examples/) | 5 scripts que construyen la red dentro de TouchDesigner | arrastrar el `.py` a la red y *Run Script* |

Cada carpeta y cada ejemplo tiene su `README.md` con la instalación y los
comandos.  Sin cámara, `python osc_simulator.py` manda datos inventados pero
con el formato exacto de la app, para todos los detectores; `make test` en
cada carpeta usa eso (y, para Processing y openFrameworks, compila los
ejemplos contra unas clases sustitutas en `tests/stubs/`, sin necesidad de
tener Processing ni openFrameworks instalados).

# Como usar estos ejemplos

Para comenzar, es indispensable tener instalado Python y un entorno con Mediapipe. Pueden seguir la guía oficial de instalación de Mediapipe [https://developers.google.com/mediapipe/solutions/setup_python](https://developers.google.com/mediapipe/solutions/setup_python)

Estos ejemplos de scripts de python tienen tres versiones: una para reconocimiento de nodos de cara (face mesh), una para reconocimiento de nodos del cuerpo (pose landmark) y una para reconocimiento de nodos de mano (hand landmarks).

Tomando como ejemplo el reconocimiento de nodos del cuerpo:

Se ejecutan dos procesos, uno en python para hacer el reconocimiento y otro en lo que sea que vaya a recibir los mensajes OSC, por ejemplo, el sketch de Processing que está en este repositorio:

```python ejemplo-pose-landmark-osc.py```

Los mensajes OSC que van desde el script en python contienen una cantidad de argumentos igual a la cantidad de nodos detectados. Es decir, en el caso del *ejemplo-pose-landmark-osc.py*, se reconocen hasta 32 nodos, según ![Pose Landmarks list](https://developers.google.com/static/mediapipe/images/solutions/pose_landmarks_index.png). Entonces, el script mandará un mensaje con 32 argumentos, cada uno conteniendo los tres valores (*x*, *y*, *z*) de posición para ese nodo, separados con espacio. Los valores están normalizados, es decir, valen entre 0 y 1.

En el ejemplo de Processing, estos valores se desglosan y se agregan a un PVector, para después dibujarse en pantalla.
