/**
 * oscP5parsing by andreas schlegel
 * example shows how to parse incoming osc messages "by hand".
 * it is recommended to take a look at oscP5plug for an
 * alternative and more convenient way to parse messages.
 * oscP5 website at http://www.sojamo.de/oscP5
 */

import oscP5.*;
import netP5.*;

OscP5 oscP5;

ArrayList <PVector>puntos;

void setup() {
  size(800,800);
  frameRate(25);
  oscP5 = new OscP5(this,5005);
  
  puntos = new ArrayList<PVector>();
}

void draw() {
  background(0);
  
  for (int i = 0; i < puntos.size(); i++) {
    PVector p = puntos.get(i);
    //ellipse(p.x, p.y, 5, 5);
    text(i, p.x*width, p.y*height); // dibujo el número de nodo en la posición del nodo... hay que tener en cuenta que los valores vienen normalizados (entre 0 y 1).
  }
}


void oscEvent(OscMessage theOscMessage) {
  
  puntos = new ArrayList<PVector>();
  
  for (int i = 0; i < theOscMessage.arguments().length; i++) {
    String nodoPos = theOscMessage.get(i).stringValue();
    String[]posicionesTmp = nodoPos.split("[ ]");
    PVector pos = new PVector(float(posicionesTmp[0]), float(posicionesTmp[1]));
    puntos.add(pos);
  } 

}
