import sys
import argparse
import os
import numpy as np
import live
import csv
from keras.models import load_model

MODEL_PATH = "../model.h5"

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
				if features_int.shape[0] == live.STEPS:
					prediction = model.predict(np.array([features_int]))
					file_results.write("{} {}\n".format(csv_name.split("/")[-1], np.argmax(prediction)))
					features_int = np.empty((0, 128))
			

