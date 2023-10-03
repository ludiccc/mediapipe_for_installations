from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import BlockingOSCUDPServer


def print_handler(address, *args):
    print(f"{address}: {args}")


def default_handler(address, *args):
    print(f"DEFAULT {address}: {args}")


def gesture_handler(address, *args):
    #print(f"GESTURE {address}: {args}")    
    gesto, score= args[0].split()
    print("Gesto", gesto, "score:",score)



def landmarks_handler(address, *args):
    print(f"LANDMARKS {address}: {args}")    

dispatcher = Dispatcher()
dispatcher.map("/gesture", gesture_handler)
dispatcher.map("/landmarks", landmarks_handler)
dispatcher.set_default_handler(default_handler)

ip = "127.0.0.1"
port = 5005

print("Ejemplo de recepción de paquetes OSC", ip, port)

server = BlockingOSCUDPServer((ip, port), dispatcher)
server.serve_forever()  # Blocks forever