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

def convert_to_speed(val, dpi=3200, refresh=125):
    return val/dpi * 0.0254 * refresh # 0.0254 is inches to meter

def cb_fct(_, dx, dy, __):
    dx = convert_to_speed(dx)
    dy = convert_to_speed(dy)
    arr.append(dx)
    arr.append(dy)
    message = str(dx) + " " + str(dy)
    sock.sendto(bytes(message, "utf-8"), ('127.0.0.1', 12345))
    sys.stdout.flush()
    print(max(arr))

pdev.setCallback(cb_fct)
while True:
    time.sleep(0.01)