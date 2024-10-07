from pynput.mouse import Controller
import keyboard
import socket 

mouse = Controller()
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

last_x = None 
last_y = None 

# Max readings i was getting was 140 

buffer = []
while(True):
    if keyboard.read_key() == "a":
    if last_y is not None:
        if mouse.position[0] - last_x != 0 or mouse.position[1] - last_y != 0:
            buffer.append(mouse.position[0] - last_x)
            buffer.append(mouse.position[1] - last_y)
            message = str(mouse.position[0] - last_x) + " " + str(mouse.position[1] - last_y)
            print(message)
            sock.sendto(bytes(message, "utf-8"), ('127.0.0.1', 12345))
    last_x = mouse.position[0]
    last_y = mouse.position[1] 

# -*- coding: utf-8 -*-

# env /usr/bin/arch -x86_64 /bin/zsh

# import libpointing

# try:
#     print("Available pointing devices:")
#     devices = libpointing.getPointingDevices()
#     for device in devices:
#         print(f"- {device}")
    
#     print("\nTrying to initialize a device:")
#     device = libpointing.PointingDevice("any:")
#     print("Device initialized successfully")
#     print(f"Device URI: {device.getURI()}")
# except Exception as e:
#     print(f"Error: {e}")