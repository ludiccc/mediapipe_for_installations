import oscP5.*;
import netP5.*;

OscP5 oscP5;

ArrayList <PVector>puntos;

void setup() {
  size(800,800,P3D);
  frameRate(25);
  oscP5 = new OscP5(this,5005);
  
  puntos = new ArrayList<PVector>();
}

void draw() {
  background(0);
  
  for (int i = 0; i < puntos.size(); i++) {
    PVector p = puntos.get(i);
    //ellipse(p.x, p.y, 5, 5);
    pushMatrix();
    translate(p.x*width, p.y*height, p.z*-100);
    text(i, 0, 0); // dibujo el número de nodo en la posición del nodo... hay que tener en cuenta que los valores vienen normalizados (entre 0 y 1).
    popMatrix();
  }
}


void oscEvent(OscMessage theOscMessage) {
  
  puntos = new ArrayList<PVector>();
  
  for (int i = 0; i < theOscMessage.arguments().length; i++) {
    String nodoPos = theOscMessage.get(i).stringValue();
    String[]posicionesTmp = nodoPos.split("[ ]");
    PVector pos = new PVector(float(posicionesTmp[0]), float(posicionesTmp[1]), float(posicionesTmp[2]));
    puntos.add(pos);
  } 

}
