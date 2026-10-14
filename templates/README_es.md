# Plantillas — apariencia de la ventana de `mediapipe_osc.py`

[English version](README.md)

Una plantilla controla solamente cómo se dibuja la ventana: panel, controles,
imagen de cámara, landmarks y texto de estado. No cambia la detección ni los
mensajes OSC. Elegila por nombre o por ruta:

```bash
python mediapipe_osc.py --template minimal
python mediapipe_osc.py --template ruta/a/mi_plantilla.py
```

Mientras la app está en marcha, `t` rota entre las plantillas disponibles y
`r` vuelve a cargar el archivo actual; esto permite editar una plantilla sin
reiniciar la instalación.

## Ejemplos incluidos

| Plantilla | Aspecto |
|---|---|
| `classic` | aspecto predeterminado oscuro |
| `minimal` | panel blanco, tipografía fina y landmarks discretos |
| `ascii` | video y controles como una grilla de caracteres ASCII |
| `baroque` | dorado sobre bordó, marcos ornamentados y sepia |

## Crear una plantilla propia

Copiá una de las plantillas de esta carpeta y definí una clase `Template` que
herede de `Theme`:

```python
from mp_osc.themes import Theme


class Template(Theme):
    bg = (200, 120, 255)  # colores BGR, como en OpenCV
    panel_side = "left"
```

Los atributos más útiles son `panel_width`, `row`, `list_row`, `font`,
`font_small`, los colores y `panel_side`. Se pueden redefinir métodos de
dibujo como `draw_toggle`, `draw_radio`, `draw_button`, `draw_field`,
`draw_list`, `draw_stepper`, `draw_section`, `draw_status`, `draw_preview` y
`draw_banner`. Consultá la [referencia completa en inglés](README.md) y los
archivos de ejemplo para ver sus firmas y parámetros.

Una plantilla debe ser rápida: se dibuja en cada cuadro. Si falla al cargarse,
la aplicación conserva la plantilla anterior y muestra el error.
