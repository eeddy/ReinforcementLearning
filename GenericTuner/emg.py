import libemg
import torch
from generic_tf_tuner import * 
import numpy as np

WINDOW_SIZE = 40
WINDOW_INC  = 5

class EMG(InputDevice):
    def __init__(self):
        super().__init__(0, 128, in_shape=1, out_shape=1, port=12345, name='emg')
    
    def run(self):
        p, smi = libemg.streamers.myo_streamer()
        odh = libemg.data_handler.OnlineDataHandler(smi)
        odh.start_listening()
        
        classifier = libemg.emg_predictor.EMGClassifier(None)
        classifier.model = torch.load('CNN_final.model', map_location=torch.device('cpu'))
        o_classifier = libemg.emg_predictor.OnlineEMGClassifier(classifier, 40, 5, odh, None, std_out=False, smm=False)
        o_classifier.run(block=True)


actions = {'low': 0, 'high': 20}

def p_func(x):
    # Creating Linear Function 
    return x/100 # We will never get to 100% MAV

if __name__ == "__main__":
    actions = {'low': 0, 'high': 5}
    pretrain = {'func': p_func, 'epochs': 10, 'samples': 100000}

    mouse = EMG()
    tuner = TFEnvironment(mouse, actions, pretrain=pretrain, timesteps=20000, emg=True)
    tuner.run('logs\emg\emg_8000_steps.zip')