# Sonido cuadrafónico: ejercicios de Juli Rojas

Ejercicios progresivos para recibir posiciones de Kinect por OSC y usarlas
para reproducir y modificar sonidos. Los comentarios y mensajes de los
programas están en castellano.

## Preparación

1. Instala Processing con las librerías `kinect4WinSDK`, `oscP5` y `netP5`.
   La librería Kinect del sketch original puede requerir Windows y sus
   controladores.
2. Deja despejada la zona bajo la Kinect. Abre
   `ejemplo_juli_rojas_2/ejemplo_juli_rojas_2.pde` en Processing y ejecútalo.
   Presiona **C** para capturar el fondo vacío. **Q** y **W** bajan y suben la
   sensibilidad de la sustracción de fondo.
3. Instala Python 3, las dependencias de `requirements.txt` y `ffmpeg`.
   `ffmpeg` debe poder ejecutarse desde la terminal.
4. Copia los sonidos MP3 o WAV a `sonidos/`. El programa carga el listado al
   inicio y decodifica cada audio al asignarlo, para no mantener toda la
   biblioteca descomprimida en memoria.

```bash
python -m pip install -r requirements.txt
python 1_recepcion_simple.py
```

Processing envía a `127.0.0.1:12000`; los programas de Python escuchan ese
puerto por defecto. Si cambias el destino o el puerto, usa `--puerto` en el
script Python. Para mandar OSC a otro equipo, cambia la dirección `destino`
del sketch.

## Mensaje OSC

Processing envía un mensaje por cada grupo de píxeles detectado:

```text
/kinect/person  id  x  y  velocidad
```

`id` es un entero. `x`, `y` y `velocidad` son valores entre 0 y 1. X está
invertida para conservar la orientación del sketch original; Y aumenta desde
el frente hacia el fondo. El sketch busca hasta ocho grupos conectados de al
menos 1200 píxeles y conserva sus IDs mientras los centroides sigan próximos.
Al dejar de recibir posiciones por un segundo, Python considera que esa
persona salió del campo.

## Recorrido de los ejemplos

| Script | Paso que muestra |
| --- | --- |
| `1_recepcion_simple.py` | Muestra en la terminal los IDs, posiciones y velocidades OSC. |
| `2_play_sonido.py` | Reproduce un archivo sin Kinect. |
| `3_pan_estereo.py` | Usa X para ubicar un sonido a izquierda o derecha. |
| `4_movimiento_cuadrafonico.py` | Reproduce un sonido con su movimiento espacial asignado. |
| `5_efectos_por_movimiento.py` | Mapea X a pitch, Y a filtro/reverb y velocidad a saturación. |
| `6_sonido_al_entrar.py` | Una nueva persona inicia una voz. |
| `7_siguiente_sonido_aleatorio.py` | Al terminar un sonido, asigna otro que no esté sonando. |
| `8_hasta_cuatro_personas.py` | Mantiene como máximo cuatro voces para distintas personas. |
| `9_efectos_colectivos.py` | Las personas extra cambian la reverb y la saturación sin sumar voces. |
| `10_instalacion_juli_rojas.py` | Une la asignación aleatoria, hasta cuatro voces, efectos y salida seleccionable. |

## Simular el seguimiento OSC

`test_send_osc.py` envía posiciones de prueba al mismo protocolo que Processing.
Cada estado dura cinco segundos: entra A, sale, A vuelve con un ID nuevo, entra
B, sale A, entran C/D/E hasta llegar a cuatro, y luego salen B/C/D/E de a una.
El ciclo vuelve a empezar con IDs nuevos.

En una terminal ejecuta el receptor o la instalación y, en otra:

```bash
python test_send_osc.py
```

El destino predeterminado es `127.0.0.1:12000`. Se puede cambiar con
`--host` y `--puerto`.

Los pasos 2 a 10 aceptan `--modo stereo` (por defecto) y
`--modo cuadrafonico`. La interfaz en modo cuadrafónico debe tener cuatro
salidas. Consulta los índices disponibles así:

```bash
python 10_instalacion_juli_rojas.py --listar-dispositivos
python 10_instalacion_juli_rojas.py --modo cuadrafonico --dispositivo 3
```

Orden de canales cuadrafónicos: 1 frente izquierda, 2 frente derecha, 3 fondo
derecha, 4 fondo izquierda. En estéreo los recorridos se reducen al eje
izquierda/derecha.

En los pasos 6 a 10, el bucle principal está escrito directamente en cada
script: allí se puede seguir la lectura de personas activas, la elección de
audios disponibles, el límite de voces y el control de efectos. Las librerías
mantienen las tareas de bajo nivel, como recibir OSC y generar los bloques de
audio.

## Sonidos y límites

El banco acepta MP3, WAV, OGG, FLAC y AIFF. Para tener cuatro voces diferentes
al mismo tiempo hacen falta al menos cuatro archivos de audio distintos. Al
reponer una voz, el sonido que acaba de terminar vuelve a estar disponible.
Los sonidos se ordenan por nombre y reciben, en ciclo, los recorridos `orbita`,
`vaiven`, `diagonal` y `persona`; se pueden editar en `cuadrafonia.py`.

El detector cenital agrupa píxeles conectados de la sustracción de fondo; si
dos siluetas se tocan pueden aparecer como una sola persona. Ajusta la
sensibilidad con Q/W y `AREA_MINIMA` en el sketch según la iluminación y la
distancia de la Kinect.
