from generic_tf_tuner import *
import serial
import serial.tools.list_ports

class SerialReader():
    def __init__(self):
        super().__init__()
        self.serialInstance = None
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def initialize_serial(self):
        ports = serial.tools.list_ports.comports()
        portList = []

        for i, port in enumerate(ports, start=1):
            portList.append(port.device)

        idx = 0 
        for i, p in enumerate(portList):
            if 'usbmodem' in p:
                idx = i 
        selected_port = portList[idx]
        print(selected_port)

        self.serialInstance = serial.Serial()
        self.serialInstance.baudrate = 9600
        self.serialInstance.port = selected_port
        self.serialInstance.open()

    def read_serial_data(self):
        buffer = []
        while True:
            try:
                if self.serialInstance.in_waiting:
                    packet = self.serialInstance.readline()
                    data = packet.decode("utf").rstrip("\n").split(" ")
                    if data[0] == "OFN":
                        dx = float(data[3]) 
                        dy = float(data[5]) * -1
                        buffer.append(dx)
                        buffer.append(dy)
                        message = str(dx) + " " + str(dy)
                        self.sock.sendto(bytes(message, "utf-8"), ('127.0.0.1', 12345))
                    else: 
                        print("not OFN formatted")
                    self.serialInstance.flush()
            except Exception as e:
                print(f"Error reading serial data: {e}")
                time.sleep(2)

class Ring(InputDevice):
    def __init__(self):
        super().__init__(0, 140, in_shape=2, out_shape=2, port=12345, name='ring')
    
    def run(self):
        sm = SerialReader()
        sm.initialize_serial()
        sm.read_serial_data()

def p_func(_):
    return np.array([0.25,0.25])

actions = {'low': 0, 'high': 5}
pretrain = {'func': p_func, 'epochs': 10, 'samples': 100000}

if __name__ == "__main__":
    mouse = Ring()
    tuner = TFEnvironment(mouse, actions, pretrain, timesteps=40000)
    tuner.run()