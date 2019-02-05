from record_and_save import RecordandSaveThread
from extract_audioset_embedding import Extractor
from keras.models import load_model
import numpy as np
import time
import argparse
import os 

STEPS = 10

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", help="Path of different model. Default is ../model.h5", default=os.path.join(os.path.dirname(os.getcwd()), "model.h5"))
    model_path = parser.parse_args().model
    labels = ["Risata", "Pianto bimbo", "Tosse", "Starnuto", "Passi, camminare", "Applauso", "Cane", "Gatto", "Acqua", "Motore", "Sveglia", "Sirena"]
    model = load_model(model_path)
    report_file = open(os.path.join(os.path.dirname(os.getcwd()), "report.txt"), "w")
    extractor = Extractor()
    record_thread = RecordandSaveThread(steps=STEPS)
    record_thread.start()
    print("Start to Record")
    array_features = np.empty((0, 128))
    confidence = 0.3
    time.sleep(1)   

    count = 0
    while True:
        time.sleep(1)
        features = extractor.extract_audioset_embedding("test{}.wav".format(count))
        count = (count + 1) % STEPS

        array_features = np.append(array_features, features, axis=0)
        
        if np.shape(array_features)[0] == STEPS:
            string_to_print = "=========================\n"
            predictions = model.predict(np.array([array_features]))[0]
            for i in range(len(predictions)):
                if predictions[i] > confidence:
                    string_to_print += "====>"
                string_to_print += " {}: {}\n".format(labels[i], predictions[i])

            print(string_to_print)
            report_file.write(string_to_print)
            array_features = np.delete(array_features, 0, 0)
