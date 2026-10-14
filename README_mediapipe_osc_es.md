# `mediapipe_osc.py` — cámara en vivo → MediaPipe / blobs de OpenCV → OSC

[English version](README_mediapipe_osc.md)

Una aplicación configurable que analiza una cámara en vivo con MediaPipe
(pose corporal, manos, gestos y cara) o con OpenCV (**Blobs**: formas vistas
desde una cámara cenital o una Kinect), y transmite el resultado como mensajes
OSC. Está pensada para instalaciones hechas con Processing, TouchDesigner,
Pure Data, Max, SuperCollider, Unity, Python u otros entornos que reciban OSC.

Respecto de los scripts `ejemplo-*.py` del repositorio, agrega:

* Todos los detectores en un proceso: `--pose --hands --face --gesture --blobs`.
* **Blobs**, para siluetas u objetos cuando los modelos de cuerpo no son
  adecuados. Cada forma recibe un ID estable y se envía como `/mp/blob/...`;
  también puede alimentar `/kinect/person` para `Sonido_Cuadrafonico`.
* Argumentos OSC numéricos, IDs estables, eventos `enter`/`leave` y velocidad.
* Formatos compatibles: `--format legacy` para sketches anteriores y
  `--format kinect` para los scripts de sonido cuadrafónico.
* Un panel de control en la ventana: detectores, fuentes, pausa, red, formatos,
  espejo y plantillas se pueden cambiar sin detener la transmisión.
* Kinect v2/v1, profundidad en metros con **Z from Kinect**, suavizado,
  límite de FPS, modo sin ventana, bundles OSC, video o imagen como fuente y
  tests automatizados.

```bash
python mediapipe_osc.py                                   # pose, webcam 0 → 127.0.0.1:5005
python mediapipe_osc.py --pose --hands --face --mirror    # imagen espejada
python mediapipe_osc.py --list-cameras                    # lista las cámaras
python mediapipe_osc.py --source 0+2                      # dos cámaras lado a lado
python mediapipe_osc.py --blobs --format kinect --port 12000
python mediapipe_osc.py --source video.mp4 --no-preview --verbose
python osc_monitor.py                                     # en otra terminal, ve lo recibido
```

Usá `python mediapipe_osc.py --help` para la referencia completa de opciones.
La ayuda de `osc_monitor.py --help` también está en ambos idiomas.

## Instalación

Se necesita Python 3.9 o posterior y estas dependencias:

```bash
pip install mediapipe opencv-python numpy python-osc
```

Los archivos de modelo `.task` deben estar en la carpeta del repositorio (ya
están incluidos) o se puede indicar otra carpeta con `--models-dir`. Para una
Kinect se requieren, además, los bindings del backend correspondiente:
`pylibfreenect2` para Kinect v2, `freenect` para v1 u OpenNI compilado en
OpenCV. Sin sensor o sin esos bindings, las demás fuentes siguen funcionando.

## La ventana y el panel

La vista previa muestra la imagen analizada y, por defecto, un panel a la
derecha. Sus cambios se aplican en vivo.

* **Detectors** activa cuerpo, manos, gestos, cara y Blobs. Activar Gestures
  también activa Hands. Al desactivar un detector se envían los `leave` de sus
  tracks. En Blobs, **Learn bg** (tecla `b`) vuelve a capturar el fondo vacío;
  los controles regulan sensibilidad y área mínima.
* **Source** permite elegir Cameras, Pause o File. La lista reúne webcams y
  Kinects; marcar varias crea una imagen compuesta, de izquierda a derecha.
  Las flechas cambian el orden y el engranaje abre el editor de recorte y
  mapeo de esa cámara. **Rescan cameras** (`c`) vuelve a buscar dispositivos.
* **Image** incluye espejo y, si la fuente tiene profundidad, **Z from Kinect**
  (`z`): sustituye la z relativa de MediaPipe por la distancia medida en metros.
* **OSC** cambia IP, puerto, formatos y pausa del envío. **Look** (`t`) rota
  las plantillas; `r` recarga la plantilla actual después de editarla.

`--no-preview` ejecuta la aplicación sin ventana. `--no-panel` conserva la
vista previa y los atajos de teclado, pero oculta el panel.

## Cámaras, Kinect y coordenadas

`--source` acepta un índice de webcam (`0`), una Kinect (`kinect` o
`kinect:1`), varias fuentes unidas con `+` (`0+kinect`), un video o una imagen.
Cuando se combinan varias fuentes, se analizan como una única imagen ancha.

Para Kinect, `--kinect-stream color` sirve para MediaPipe y `--kinect-stream
depth` usa el mapa de profundidad (cerca = brillante), conveniente para Blobs.
`auto` elige profundidad cuando solo está activo Blobs y color en los demás
casos. `--z-from-kinect` requiere una fuente Kinect —o varias Kinects—, no una
mezcla de Kinect y webcam.

Las coordenadas `x` e `y` están normalizadas entre 0 y 1, con origen arriba a
la izquierda de la imagen analizada. Con `--mirror`, x aumenta hacia la derecha
de quien mira. La `z` normal es la profundidad relativa de MediaPipe; con
**Z from Kinect** es distancia al sensor en metros. Las velocidades están en
unidades normalizadas por segundo.

## Mensajes OSC: formato `mp` (predeterminado)

Los tipos son `i` (entero), `f` (float) y `s` (texto). `<id>` es un ID estable
mientras la entidad está en escena; poses, manos, caras y blobs usan contadores
independientes. Al volver a entrar, una entidad recibe otro ID.

```
/mp/status            s:"started"|"stopped"  s:version
/mp/z_source          s:"mediapipe"|"kinect"
/mp/frame             i:frame f:time_s f:fps i:n_poses i:n_hands i:n_faces i:n_blobs

/mp/pose/enter|leave  i:id
/mp/pose/center       i:id f:x f:y f:z
/mp/pose/bbox         i:id f:xmin f:ymin f:xmax f:ymax
/mp/pose/velocity     i:id f:vx f:vy f:speed
/mp/pose/landmarks    i:id 33 × (f:x f:y f:z f:visibility)
/mp/pose/world        i:id 33 × (f:x f:y f:z)             # --world

/mp/hand/enter|leave  i:id
/mp/hand/info         i:id s:"Left"|"Right" f:score
/mp/hand/center|bbox|velocity  ...
/mp/hand/pinch        i:id f:distance
/mp/hand/gesture      i:id s:name f:score                  # --gesture
/mp/hand/landmarks    i:id 21 × (f:x f:y f:z)

/mp/face/enter|leave  i:id
/mp/face/center|bbox|velocity  ...
/mp/face/blendshapes  i:id 52 × f:value
/mp/face/landmarks    i:id 478 × (f:x f:y f:z)             # --face-landmarks

/mp/blob/enter|leave  i:id                                 # --blobs
/mp/blobs             i:count count × i:id
/mp/blob/center|bbox|velocity  ...
/mp/blob/info         i:id f:area f:angle f:length f:width
/mp/blob/contour      i:id 16 × (f:x f:y)
```

El orden por cuadro es `/mp/frame`, los `enter`, los datos de los tracks
activos y los `leave`. Al salir, la app emite `leave` para todos los tracks y
`/mp/status stopped`.

Con `--id-style address`, el ID pasa a la dirección:
`/mp/pose/3/center x y z`. `--prefix /cam1` cambia el prefijo `/mp` y
`--bundle` empaqueta los mensajes de un cuadro en un bundle OSC.

### Formatos compatibles

`--format legacy` conserva los mensajes de texto usados por los scripts
originales de Processing. `--format kinect` produce
`/kinect/person id x y speed`, esperado por `Sonido_Cuadrafonico`; suele usarse
con `--port 12000`. Con una cámara cenital, `--kinect-y image` usa la y de la
imagen; con una cámara frontal, `--kinect-y size` estima la profundidad por el
tamaño aparente. Los formatos se pueden combinar: `--format mp kinect`.

## Blobs: formas en lugar de cuerpos

`--blobs` ejecuta una cadena de OpenCV sin modelo de MediaPipe: gris → blur →
máscara de primer plano → limpieza morfológica → contornos. Cada forma dentro
del rango `--blob-min-area`/`--blob-max-area` obtiene centro, caja, área,
orientación y un contorno de 16 puntos, y pasa por el mismo tracker que los
demás detectores.

`--blob-mode background` compara con un fondo vacío aprendido; iniciá la app
sin personas en escena o presioná `b`/ **Learn bg**. `motion` usa MOG2 y los
objetos quietos terminan formando parte del fondo. `bright` y `dark` usan un
umbral fijo, útil para objetos claros/oscuros o para profundidad Kinect.
`--blob-roi x0 y0 x1 y1` limita la detección a un rectángulo normalizado.

Ejemplo de cámara cenital para el sonido cuadrafónico:

```bash
python mediapipe_osc.py --blobs --format kinect --port 12000
python Sonido_Cuadrafonico/10_instalacion_juli_rojas.py
```

## Plantillas y configuración

Todo el dibujo pasa por una plantilla (`Theme`). Están incluidas `classic`,
`minimal`, `ascii` y `baroque`; se eligen con `--template`. Una plantilla
propia es un archivo Python que define `class Template(Theme)` y cambia colores,
tipografías, geometría o los métodos `draw_*`; no modifica los detectores ni
OSC. Consultá [templates/README_es.md](templates/README_es.md) como referencia
en castellano (y [su versión en inglés](templates/README.md)).

**Save config** guarda en JSON cámaras, detectores, Blobs, OSC, imagen,
plantilla y mapeos. **Load config** los restituye en vivo. También se puede
iniciar con `--config archivo.json`.

## Receptores, tests y estructura

* `osc_monitor.py` permite inspeccionar lo que llega: `--rate` cuenta mensajes
  por dirección y `--filter /mp/blob` filtra por prefijo.
* `osc_mp_tracking_example/` y `osc_mp_blobs_example/` son sketches de
  Processing para el formato `mp`.
* `Sonido_Cuadrafonico/` recibe `/kinect/person`.

Para ejecutar los tests:

```bash
pip install pytest
python -m pytest tests/ -q
```

La app está separada en módulos: `capture.py` lee fuentes, `detectors.py` y
`blobs.py` encuentran entidades, `tracking.py` mantiene IDs, `osc_out.py`
construye y envía mensajes, y `app.py` coordina el ciclo. La explicación para
modificarla está en la [guía para estudiantes](GUIDE_mediapipe_osc_es.md).
