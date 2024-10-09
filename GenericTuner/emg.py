import libemg
import torch
from generic_tf_tuner import * 
import numpy as np

WINDOW_SIZE = 40
WINDOW_INC  = 5

class EMG(InputDevice):
    def __init__(self, odh):
        self.odh = odh
        super().__init__(0, 128, in_shape=8, out_shape=1, port=12345, name='emg')
    
    def run(self):
        num_samples = 0

        while True:
            data, counts = self.odh.get_data()
            if counts['emg'] > num_samples + WINDOW_INC:
                mav = np.mean(np.abs(data['emg'][:WINDOW_SIZE]),0)
                message = " ".join([str(m) for m in mav])
                print(message)
                self.sock.sendto(bytes(message, 'utf-8'), ('127.0.0.1', 12345))
                num_samples = counts['emg']

def p_func(_):
    return np.array([1,1])

actions = {'low': 0, 'high': 20}
pretrain = {'func': p_func, 'epochs': 10, 'samples': 100000}

if __name__ == "__main__":
    p, smi = libemg.streamers.myo_streamer()
    odh = libemg.data_handler.OnlineDataHandler(smi)
    odh.start_listening()
    
    classifier = libemg.emg_predictor.EMGClassifier(None)
    classifier.model = torch.load('CNN_final.model', map_location=torch.device('cpu'))
    o_classifier = libemg.emg_predictor.OnlineEMGClassifier(classifier, 40, 5, odh, None, std_out=False, smm=False)
    o_classifier.run(block=False)

    mouse = EMG(odh)
    tuner = TFEnvironment(mouse, actions, pretrain, timesteps=20000)
    tuner.run()