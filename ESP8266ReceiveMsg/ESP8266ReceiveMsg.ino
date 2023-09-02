/*---------------------------------------------------------------------------------------------

Ejemplo de recepción de datos de control usando OSC

--------------------------------------------------------------------------------------------- */
#include <ESP8266WiFi.h>
#include <WiFiUdp.h>
#include <OSCMessage.h>
#include <OSCBundle.h>
#include <OSCData.h>

char ssid[] = "****";  // el nombre de la red (SSID)
char pass[] = "****";  // la clave de la red

// El objeto WifiUDP es el que permite enviar y recibir paquetes
WiFiUDP Udp;
const IPAddress outIp(192,168,0,205);       // IP remota (no hace falta para actuar como receptor, sí para enviar)
const unsigned int outPort = 9999;          // puerto remoto (idem anterior)

// puerto que va a abrir el ESP8266 internamente y al cual la computadora o el sistema remoto va a tener que conectarse
const unsigned int localPort = 8888;        


OSCErrorCode error;
unsigned int ledState = LOW;              

void setup() {
  pinMode(BUILTIN_LED, OUTPUT);
  digitalWrite(BUILTIN_LED, ledState);    

  Serial.begin(115200);

  // Conectarse a la red WiFi
  Serial.println();
  Serial.println();
  Serial.print("Conectandose a... ");
  Serial.println(ssid);
  WiFi.begin(ssid, pass);

  //wait until connected
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("");

  Serial.println("WiFi conectada");
  Serial.println("Direccion IP: ");
  Serial.println(WiFi.localIP());

  Serial.println("Iniciando UDP");
  Udp.begin(localPort);
  Serial.print("Puerto local: ");
  Serial.println(Udp.localPort());

}

// Vamos a llamar a esta función cuando recibamos un mensaje "/led"
void led(OSCMessage &msg) {
  // Verificamos que el primer parámetro es de tipo int para usarlo
  if (msg.isInt(0)) {
    ledState = msg.getInt(0);
    digitalWrite(BUILTIN_LED, ledState);
    Serial.print("/led: ");
    Serial.println(ledState);
  }

  // Si hubiese otro tipo de mensaje u otro parámetro, por ejemplo, si hay dos leds...
  /*
  if (msg.isInt(1)) {
    led2State = msg.getInt(1);
    ...
  }
  */
}

void loop() {
  OSCMessage msg;
  
  // obtener los datos que vienen de la red
  int size = Udp.parsePacket();

  // Solo ejecutaremos los siguiente, si efectivamente hay datos
  if (size > 0) {
    while (size--) {
      /*
       * si el cliente estaba enviando un bundle osc
       * en vez de un mensaje, la linea siguiente deberá ser:
       * bundle.fill(Udp.read());
       */
      msg.fill(Udp.read());
    }
    
    if (!msg.hasError()) {
      /* Si el mensaje contenía la dirección "/led"
       * va a llamar a la función led definida más arriba.
       * Si no tenía esta dirección, el NodeMCU no va a hacer nada
       */
      msg.dispatch("/led", led);

      /* También se pueden definir otras direcciones 
       * como, por ejemplo, "/motor" para que llame 
       *  a las funciones correpondientes.
       */
       //msg.dispatch("/motor", motor_function);
    } else {
      error = msg.getError();
      Serial.print("error: ");
      Serial.println(error);
    }
  }
}
