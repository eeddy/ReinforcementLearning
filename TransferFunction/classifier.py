import libemg
import time
import numpy as np


class CustomOutputEMGClassifier(libemg.emg_predictor.OnlineEMGClassifier):
    def __init__(self,  offline_classifier, window_size, window_increment, online_data_handler, features, 
                 file_path = '.', file=False, smm=False, 
                 smm_items= None,
                 port=12346, ip='127.0.0.1', std_out=False, tcp=False,
                 output_format="predictions"):
        super(CustomOutputEMGClassifier, self).__init__(offline_classifier, window_size, window_increment, online_data_handler, features, 
                 file_path, file, smm, 
                 smm_items,
                 port, ip, std_out, tcp,
                 output_format)

    def write_output(self, model_input, window):
        # Make prediction
        probabilities = self.predictor.model.predict_proba(model_input)
        prediction, probability = self.predictor._prediction_helper(probabilities)
        prediction = prediction[0]
        
        message = str(prediction) +" "+ " ".join([str(i) for i in probabilities.squeeze().tolist()]) + " " + " ".join([str(i) for i in np.mean(np.abs(window['emg']),2).squeeze().tolist()]) + '\n'
        self.sock.sendto(bytes(message, 'utf-8'), (self.ip, self.port))
    

if __name__ == "__main__":
    # train
    p, smi = libemg.streamers.myo_streamer()
    odh = libemg.data_handler.OnlineDataHandler(smi)
    
    
    
    gui = libemg.gui.GUI(online_data_handler = odh,
                        args={ "online_data_handler": odh,
                        "streamer":p,
                        "media_folder": "Images/",
                        "data_folder":  f"Data/",
                        "num_reps":     5,
                        "rep_time":     2,
                        "rest_time":    1,
                        "auto_advance": True})
    gui.download_gestures([1,2,3,4,5], 'Images/')
    gui.start_gui()

    # prepare classifier
    offline_data_handler = libemg.data_handler.OfflineDataHandler()
    offline_data_handler.get_data(
        "Data/",
        [libemg.data_handler.RegexFilter("C_", "_R_", [str(i) for i in range(5)], "classes"),
         libemg.data_handler.RegexFilter("_R_", "_emg.csv", [str(i) for i in range(5)], "reps")]
    )
    
    windows, metadata = offline_data_handler.parse_windows(40, 5)

    fe = libemg.feature_extractor.FeatureExtractor()
    features = fe.extract_feature_group("LS4",windows)

    classifier = libemg.emg_predictor.EMGClassifier("LDA")
    classifier.fit({"training_features": features,
                    "training_labels": metadata["classes"]})
    
    o_classifier = CustomOutputEMGClassifier(classifier, 40, 5, odh, fe.get_feature_groups()['LS4'])
    o_classifier.run(True)

    A  = 1
