import kinect4WinSDK.*;
import oscP5.*;
import netP5.*;
import java.util.Arrays;

// Ejemplo mínimo: Kinect cenital -> detección de siluetas -> OSC.
// No contiene reproductor de audio ni salida MIDI.

final int ANCHO_KINECT = 640;
final int ALTO_KINECT = 480;
final int CANTIDAD_PIXELES = ANCHO_KINECT * ALTO_KINECT;
final int MAX_PERSONAS = 8;
final int AREA_MINIMA = 1200;
final int TIEMPO_PERDIDA_MS = 600;
final float DISTANCIA_ASOCIACION = 100;
final int PUERTO_OSC = 12000;

Kinect kinect;
OscP5 oscP5;
NetAddress destino;
PImage silueta;

float[] fondo = new float[CANTIDAD_PIXELES];
boolean[] mascara = new boolean[CANTIDAD_PIXELES];
boolean[] visitado = new boolean[CANTIDAD_PIXELES];
int[] cola = new int[CANTIDAD_PIXELES];
boolean fondoCapturado = false;
int sensibilidad = 15;
int siguienteId = 1;
ArrayList<PersonaTrack> personas = new ArrayList<PersonaTrack>();

int minX = 150;
int maxX = 490;
int minY = 100;
int maxY = 380;

void setup() {
  size(1280, 480);
  frameRate(30);
  kinect = new Kinect(this);
  silueta = createImage(ANCHO_KINECT, ALTO_KINECT, RGB);

  // Processing escucha localmente en 9000 y envía las posiciones a Python.
  oscP5 = new OscP5(this, 9000);
  destino = new NetAddress("127.0.0.1", PUERTO_OSC);
}

void draw() {
  background(24);
  PImage profundidad = kinect.GetDepth();
  if (profundidad == null || profundidad.width != ANCHO_KINECT) {
    fill(255);
    text("Esperando datos de Kinect...", 20, 30);
    return;
  }

  profundidad.loadPixels();
  if (!fondoCapturado) {
    image(profundidad, 0, 0, ANCHO_KINECT, ALTO_KINECT);
    fill(255, 220, 80);
    textSize(16);
    text("Deja despejado el espacio y presiona C para calibrar", 20, 25);
    return;
  }

  silueta.loadPixels();
  for (int i = 0; i < CANTIDAD_PIXELES; i++) {
    float diferencia = abs(brightness(profundidad.pixels[i]) - fondo[i]);
    mascara[i] = diferencia > sensibilidad;
    silueta.pixels[i] = mascara[i] ? color(255) : color(0);
  }
  silueta.updatePixels();

  ArrayList<Componente> detecciones = detectarComponentes();
  actualizarPersonas(detecciones);

  image(profundidad, 0, 0, ANCHO_KINECT, ALTO_KINECT);
  image(silueta, ANCHO_KINECT, 0, ANCHO_KINECT, ALTO_KINECT);
  enviarPersonas();
  dibujarPersonas();

  fill(0, 0, 0, 175);
  noStroke();
  rect(0, ALTO_KINECT - 42, width, 42);
  fill(255);
  textSize(14);
  text("OSC -> 127.0.0.1:" + PUERTO_OSC + "   Personas: " + contarVisibles()
       + "   Sensibilidad: " + sensibilidad, 14, ALTO_KINECT - 17);
  text("C: calibrar fondo   Q/W: sensibilidad", 660, ALTO_KINECT - 17);
}

ArrayList<Componente> detectarComponentes() {
  ArrayList<Componente> resultado = new ArrayList<Componente>();
  Arrays.fill(visitado, false);

  for (int inicio = 0; inicio < CANTIDAD_PIXELES; inicio++) {
    if (!mascara[inicio] || visitado[inicio]) continue;

    int cabeza = 0;
    int finalCola = 0;
    cola[finalCola++] = inicio;
    visitado[inicio] = true;
    int cantidad = 0;
    long sumaX = 0;
    long sumaY = 0;

    while (cabeza < finalCola) {
      int indice = cola[cabeza++];
      int x = indice % ANCHO_KINECT;
      int y = indice / ANCHO_KINECT;
      cantidad++;
      sumaX += x;
      sumaY += y;

      if (x > 0 && agregarVecino(indice - 1, finalCola)) finalCola++;
      if (x < ANCHO_KINECT - 1 && agregarVecino(indice + 1, finalCola)) finalCola++;
      if (y > 0 && agregarVecino(indice - ANCHO_KINECT, finalCola)) finalCola++;
      if (y < ALTO_KINECT - 1 && agregarVecino(indice + ANCHO_KINECT, finalCola)) finalCola++;
    }

    if (cantidad >= AREA_MINIMA) {
      resultado.add(new Componente((float)sumaX / cantidad,
                                   (float)sumaY / cantidad, cantidad));
    }
  }

  resultado.sort((a, b) -> b.area - a.area);
  if (resultado.size() > MAX_PERSONAS) {
    resultado = new ArrayList<Componente>(resultado.subList(0, MAX_PERSONAS));
  }
  return resultado;
}

boolean agregarVecino(int indice, int posicion) {
  if (mascara[indice] && !visitado[indice]) {
    visitado[indice] = true;
    cola[posicion] = indice;
    return true;
  }
  return false;
}

void actualizarPersonas(ArrayList<Componente> detecciones) {
  int ahora = millis();
  for (PersonaTrack persona : personas) persona.detectadaEsteCuadro = false;

  for (Componente deteccion : detecciones) {
    PersonaTrack mejor = null;
    float menorDistancia = DISTANCIA_ASOCIACION;
    for (PersonaTrack persona : personas) {
      if (persona.detectadaEsteCuadro || ahora - persona.ultimaVista > TIEMPO_PERDIDA_MS) continue;
      float distancia = dist(persona.x, persona.y, deteccion.x, deteccion.y);
      if (distancia < menorDistancia) {
        menorDistancia = distancia;
        mejor = persona;
      }
    }

    if (mejor == null) {
      if (personas.size() >= MAX_PERSONAS) continue;
      mejor = new PersonaTrack(siguienteId++, deteccion.x, deteccion.y, ahora);
      personas.add(mejor);
    } else {
      float distancia = dist(mejor.x, mejor.y, deteccion.x, deteccion.y);
      mejor.velocidad = constrain(distancia / 16.0, 0, 1);
      mejor.x = lerp(mejor.x, deteccion.x, 0.35);
      mejor.y = lerp(mejor.y, deteccion.y, 0.35);
      mejor.ultimaVista = ahora;
    }
    mejor.detectadaEsteCuadro = true;
  }

  for (int i = personas.size() - 1; i >= 0; i--) {
    if (ahora - personas.get(i).ultimaVista > TIEMPO_PERDIDA_MS) personas.remove(i);
  }
}

void enviarPersonas() {
  for (PersonaTrack persona : personas) {
    if (!persona.detectadaEsteCuadro) continue;

    // Se conserva la inversión horizontal del sketch original.
    float x = constrain(map(persona.x, minX, maxX, 1, 0), 0, 1);
    float y = constrain(map(persona.y, minY, maxY, 0, 1), 0, 1);
    OscMessage mensaje = new OscMessage("/kinect/person");
    mensaje.add(persona.id);
    mensaje.add(x);
    mensaje.add(y);
    mensaje.add(persona.velocidad);
    oscP5.send(mensaje, destino);
  }
}

void dibujarPersonas() {
  pushStyle();
  for (PersonaTrack persona : personas) {
    if (!persona.detectadaEsteCuadro) continue;
    noStroke();
    fill(50, 235, 130);
    ellipse(ANCHO_KINECT + persona.x, persona.y, 22, 22);
    fill(0);
    textAlign(CENTER, CENTER);
    text(persona.id, ANCHO_KINECT + persona.x, persona.y);
  }
  popStyle();
}

int contarVisibles() {
  int total = 0;
  for (PersonaTrack persona : personas) {
    if (persona.detectadaEsteCuadro) total++;
  }
  return total;
}

void keyPressed() {
  if (key == 'c' || key == 'C') {
    PImage profundidad = kinect.GetDepth();
    if (profundidad == null) return;
    profundidad.loadPixels();
    for (int i = 0; i < CANTIDAD_PIXELES; i++) {
      fondo[i] = brightness(profundidad.pixels[i]);
    }
    fondoCapturado = true;
    personas.clear();
    println("Fondo calibrado. Enviando posiciones OSC al puerto " + PUERTO_OSC + ".");
  }
  if (key == 'q' || key == 'Q') sensibilidad = max(2, sensibilidad - 1);
  if (key == 'w' || key == 'W') sensibilidad = min(100, sensibilidad + 1);
}

class Componente {
  float x;
  float y;
  int area;

  Componente(float x, float y, int area) {
    this.x = x;
    this.y = y;
    this.area = area;
  }
}

class PersonaTrack {
  int id;
  float x;
  float y;
  float velocidad = 0;
  int ultimaVista;
  boolean detectadaEsteCuadro = true;

  PersonaTrack(int id, float x, float y, int ahora) {
    this.id = id;
    this.x = x;
    this.y = y;
    this.ultimaVista = ahora;
  }
}
