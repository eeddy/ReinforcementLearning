from generic_tf_tuner import *
from libpointing import PointingDevice
from libpointing import PointingDeviceManager
import sys

class Mouse(InputDevice):
    def __init__(self):
        super().__init__(0, 140, in_shape=2, out_shape=2, port=12345, name='mouse')
    
    def run(self):
        pdev = PointingDevice(b"any:")
        pdev.setCallback(self._cb_fct)
        while True:
            time.sleep(0.01)
    
    def _cb_fct(self, _, dx, dy, __):
        message = str(dx) + " " + str(dy)
        self.sock.sendto(bytes(message, "utf-8"), ('127.0.0.1', self.port))
        sys.stdout.flush()

def p_func(x):
    return np.array([4,4])

actions = {'low': 0, 'high': 20}
pretrain = {'func': p_func, 'epochs': 10, 'samples': 100000}

if __name__ == "__main__":
    mouse = Mouse()
    tuner = TFEnvironment(mouse, actions, pretrain)
    tuner.run()