import serial 
import serial.tools.list_ports
import time
import socket
import numpy as np 
import random
import threading
import math
from multiprocessing import Process

def start_process():
    sm = SerialReader()
    sm.initialize_serial()
    sm.read_serial_data()

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
        tot_x = 0
        tot_y = 0
        send_time = time.time()
        while True:
            try:
                if time.time() - send_time >= 0.033333333: 
                    dx = int(tot_x)
                    dy = int(tot_y)
                    message = str(dx) + " " + str(dy)
                    self.sock.sendto(bytes(message, "utf-8"), ('127.0.0.1', 12345))
                    send_time = time.time()
                    tot_x = 0 
                    tot_y = 0
                if self.serialInstance.in_waiting:
                    packet = self.serialInstance.readline()
                    data = packet.decode("utf").rstrip("\n").split(" ")
                    if data[0] == "OFN":
                        tot_x += float(data[3]) 
                        tot_y += float(data[5]) 
                    else: 
                        print("not OFN formatted")
                    self.serialInstance.flush()

            except Exception as e:
                print(f"Error reading serial data: {e}")
                time.sleep(2)

if __name__ == "__main__":
    start_process()