# Cómo funciona `mediapipe_osc.py` — guía para estudiantes

[English version](GUIDE_mediapipe_osc.md)

Esta guía acompaña el código desde la captura de una imagen hasta el envío de
OSC. Para opciones y uso cotidiano, consultar el
[manual de uso](README_mediapipe_osc_es.md).

## La idea general

Una cámara, Kinect, video o imagen produce cuadros. En cada cuadro los
detectores encuentran los 33 landmarks del cuerpo, los 21 de cada mano, los
478 de una cara o las formas (*blobs*) que se destacan del fondo. El tracker
decide cuáles detecciones son la misma entidad que en el cuadro anterior y les
da un ID. Finalmente se generan mensajes OSC como
`/mp/pose/center 3 0.52 0.41 -0.10` y se envían por UDP a Processing,
TouchDesigner, Pd, Max, SuperCollider, Unity u otro programa. La ventana deja
ver y cambiar la configuración.

| Término | Significado en esta app |
|---|---|
| **frame/cuadro** | Una imagen BGR de OpenCV, un `numpy array` `(alto, ancho, 3)`. |
| **landmark** | Un punto de un modelo MediaPipe: `x`, `y`, `z`; pose añade visibilidad. |
| **detection** | Algo encontrado en un cuadro, sin memoria ni identidad. |
| **track** | Una detección seguida en el tiempo: tiene ID, centro y velocidad. |
| **source/fuente** | Un `FrameSource`: webcam, Kinect, composición, video o imagen. |
| **blob** | Una forma conectada de una máscara binaria de OpenCV. |
| **template/plantilla** | Archivo Python que define el aspecto de la ventana. |

## El recorrido de un cuadro

```
FrameSource → Analyzer → Tracker → MessageBuilder → OscSender → UDP
 capture.py   detectors/  tracking.py   osc_out.py     osc_out.py
              blobs.py
                  └── (Z from Kinect: depth.py)
Tracker ──→ Theme.draw_preview → ventana + ControlPanel
```

`App.step()` en `mp_osc/app.py` aplica cambios pedidos desde el panel, lee un
cuadro, ejecuta los detectores, aplica profundidad Kinect si corresponde,
actualiza los trackers, envía OSC y dibuja la vista previa. Separar esos pasos
en módulos permite entenderlos y probarlos de manera independiente.

## Módulos principales

* **`config.py`**: `Config` reúne opciones y `build_parser()` define la línea
  de comandos. Para crear una opción, agregar el atributo, el flag y usarlo.
* **`capture.py`**: define `FrameSource` y abre webcam, Kinect, video, imagen
  o composiciones. Las webcams conservan solo el cuadro más reciente para no
  acumular demora.
* **`cameras.py`**: busca dispositivos para la lista del panel en segundo plano.
* **`detectors.py`**: crea los Tasks de MediaPipe y convierte resultados a
  arrays de NumPy. **`blobs.py`** produce máscaras y contornos con OpenCV.
* **`mapping.py`**: recorta y transforma cada cámara con cuadriláteros de
  entrada/salida; incluye el editor y JSON compatible con tpf-tools.
* **`depth.py`**: toma la profundidad Kinect cerca de cada landmark y escribe
  esa distancia en metros en z.
* **`tracking.py`**: asocia detecciones por cercanía, crea `enter`/`leave` y
  suaviza velocidades y landmarks.
* **`osc_out.py`**: transforma tracks en mensajes y los envía por UDP, en los
  formatos `mp`, `legacy` y `kinect`.
* **`themes.py`, `preview.py`, `ui.py`**: dibujan la ventana y el panel. Una
  plantilla hereda de `Theme` y redefine solo el aspecto visual.
* **`app.py`**: conecta todo; es el mejor archivo para leer primero.

## Hilos y una regla importante

Con vista previa hay dos hilos: el de análisis llama repetidamente a `step()`;
el principal muestra la imagen y atiende teclado y mouse con `cv2.waitKey`.
Esto mantiene el panel ágil aunque el modelo sea lento.

Los objetos MediaPipe y la fuente de video deben tocarse solamente desde el
hilo de análisis. Por eso el panel encola cambios con
`_on_analysis_thread()` y `step()` los aplica entre dos cuadros. Sin vista
previa hay un solo hilo y esa cola se ejecuta de inmediato.

## Coordenadas, tiempo e IDs

`x` e `y` se normalizan a la imagen analizada. Con cámaras lado a lado, se
normalizan a toda la composición. La z usual de MediaPipe es relativa; la de
Kinect está en metros y vale 0 cuando no hay lectura. Las fuentes en vivo usan
reloj monotónico y los videos su propia tasa de cuadros. Cada tracker es
independiente: el ID 1 de una mano no está relacionado con el ID 1 de una pose.

## Recetas para extender la app

**Agregar un mensaje OSC.** En `MessageBuilder._mp_frame()` de `osc_out.py`,
agregar el mensaje, documentarlo en el manual y crear una prueba en
`tests/test_osc_out.py`.

**Agregar un control al panel.** Añadir el atributo a `Config` y el flag. Si
el cambio toca un detector o fuente, crear un método `set_*` en `App`; luego
agregar el widget en `ControlPanel._build()`. Los tests pueden simularlo con
`panel.click("Etiqueta")`.

**Agregar un detector.** Crear una dataclass con `landmarks` y `center`,
ejecutarla desde `Analyzer.analyze()`, darle un `Tracker` en `App`, mensajes en
`MessageBuilder` y dibujo en `preview`. `BlobDetector` es el ejemplo completo
más pequeño.

**Cambiar la apariencia.** Copiar `templates/minimal.py`, modificar colores o
un método `draw_*`, ejecutar con `--template archivo.py` y presionar `r` tras
cada guardado. La plantilla no debería tener lógica de detección ni OSC.

**Trabajar sin hardware.** Usar `--source foto.jpg` o `--source clip.mp4`.
Los tests usan imágenes y videos sintéticos para no requerir cámara.

## Tests y ejercicios

```bash
python -m pytest tests/ -q
```

Hay pruebas unitarias de tracking, mensajes, Blobs, profundidad, captura,
cámaras y Kinect, además de pruebas integradas de la app y del panel. Al
agregar una función, sumar una prueba pequeña que no dependa de una cámara real.

1. Enviar la altura de cada persona como `/mp/pose/height id h` y usarla para
   cambiar el tamaño de un círculo en Processing.
2. Añadir un stepper de suavizado ligado a `cfg.smooth`.
3. Diseñar una plantilla que asigne un color distinto a cada ID.
4. Con cámara cenital y Blobs, crear una voz por persona en Pd o SuperCollider,
   paneada por `x`; probar después `--z-from-kinect` para controlar un filtro.
