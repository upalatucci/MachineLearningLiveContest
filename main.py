from record_and_save import RecordandSaveThread
from extract_audioset_embedding import Extractor
from keras.models import load_model
import numpy as np
import time
STEPS = 10

if __name__ == '__main__':
    labels = ["Risata", "Pianto bimbo", "Tosse", "Starnuto", "Passi, camminare", "Applauso", "Cane", "Gatto", "Acqua", "Motore", "Sveglia", "Sirena"]
    model = load_model("../tot_model_1024_04_256.h5")
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
        count += 1

        if count == STEPS:
            count = 0
        array_features = np.append(array_features, features, axis=0)

        if np.shape(array_features)[0] == STEPS:
            predictions = model.predict(np.array([array_features]))[0]
            print("Prediction: {}".format(predictions))

            for i in range(len(predictions)):
                if predictions[i] > confidence:
                    print("Label: {}".format(labels[i]))
            array_features = np.empty((0, 128))
