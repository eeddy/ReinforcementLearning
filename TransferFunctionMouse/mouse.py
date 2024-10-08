from libpointing import PointingDevice, DisplayDevice, TransferFunction
from libpointing import PointingDeviceManager, PointingDeviceDescriptor
import time
import sys
import socket

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
arr = []

pm = PointingDeviceManager()
PointingDevice.idle(100)

pdev = PointingDevice(b"any:")

def cb_fct(_, dx, dy, __):
    message = str(dx) + " " + str(dy)
    sock.sendto(bytes(message, "utf-8"), ('127.0.0.1', 12345))
    sys.stdout.flush()

pdev.setCallback(cb_fct)
while True:
    time.sleep(0.01)