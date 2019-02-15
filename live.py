"""
    At the start, the window have size 0. Every second, the programm read the wav file create from the ReadandSaveThread
    and convert the audio information into features using the Extractor.

    So, when the window have the minimum size to be processed from the CustomMulticlassifier, the prediction begin.
    Every second we have the prediction and the window became to be larger.
    With a larger window the CustomMulticlassifier use a model with a
    time_steps bigger and also can give different predictions to have a higher accuracy. But at the same time, with
    a very big window, the final prediction is influensed by a very old events, so with a smaller window we give a
    prediction only for earliest events.

    In the live can be setted the size of the maximum window to use for the prediction (-w) and also the path
    of the configuration to use to build CustomMulticlassifier.
    The default value of window is the steps of the models of the multiclassifier with the maximum number of steps.
"""
import argparse
import os
import time

import numpy as np
from custom_multiclassifier import CustomMulticlassifier
from extract_audioset_embedding import Extractor
from record_and_save import RecordandSaveThread

labels = {
    16: "Risata",
    23: "Pianto bimbo",
    47: "Tosse",
    49: "Starnuto",
    53: "Passi, camminare",
    67: "Applauso",
    74: "Cane",
    81: "Gatto",
    288: "Acqua",
    343: "Motore",
    395: "Sveglia",
    396: "Sirena"
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--configuration_models", help="Path of configuration models. Default is configuration.json", default=os.path.join(os.getcwd(), "configuration.json"))
    configuration_path = parser.parse_args().configuration_models
    multiclass = CustomMulticlassifier(configuration_path)
    parser.add_argument("-w", "--window", help="Maximum window to consider for the prediction.", default=multiclass.max_steps)
    size_window = parser.parse_args().window

    report_file = open(os.path.join(os.path.dirname(os.getcwd()), "report.txt"), "w")
    extractor = Extractor()
    record_thread = RecordandSaveThread(steps=size_window)
    record_thread.start()
    print("Start to Record")
    array_features = np.empty((0, 128))
    confidence = 0.3

    time.sleep(2)

    count = 0
    while True:
        time.sleep(1)
        features = extractor.extract_audioset_embedding("test{}.wav".format(count))
        array_features = np.append(array_features, features, axis=0)
        if multiclass.min_steps < array_features.shape[0]:
            prediction, accuracy = multiclass.predict(array_features)
            print("{} - {}".format(labels[prediction], accuracy))

        if np.shape(array_features)[0] == size_window:
            array_features = np.delete(array_features, 0, 0)

        count = (count + 1)% size_window
