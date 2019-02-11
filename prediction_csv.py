import sys
import argparse
import os
import numpy as np
import csv
from keras.models import load_model

MODEL_PATH = "../model.h5"

MODEL_STEPS = 10
class_to_consider = [16, 23, 47, 49, 53, 67, 74, 81, 288, 343, 395, 396]
"""

Predizione utilizzando argmax!
Perchè model.predict ritorna un array di 12 elementi che rappresentano le probabilità di ogni singola label. 

"""
if __name__ == "__main__":
	assert(len(sys.argv) == 3)
	csv_path = sys.argv[1]
	file_results = open(sys.argv[2], "w")
	csv_names = [os.path.join(csv_path, f) for f in os.listdir(csv_path)]

	model = load_model(MODEL_PATH)
	for csv_name in csv_names:
		features_int = np.empty((0, 128))
		if(csv_name.split(".")[-1] == "csv"):
			csv_reader = csv.reader(open(csv_name), delimiter=",")
			for row in csv_reader:
				second_array = np.array([int(x) for x in row])
				features_int = np.vstack((features_int, second_array))
				if features_int.shape[0] == MODEL_STEPS:
					prediction = model.predict(np.array([features_int]))
					print(np.argmax(prediction))
					file_results.write("{} {}\n".format(csv_name.split("/")[-1], class_to_consider[np.argmax(prediction)]))
					features_int = np.empty((0, 128))
			

