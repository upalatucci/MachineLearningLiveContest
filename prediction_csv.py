"""

    The program read all the rows of the csv file and after give a prediction for that file.
    To work it need two argument that are the path of the directory with all the csv files and the path of the file in which
    write all the results. All the argument can be see using -h argument.

    The format of the results is:

    {filename} {class predicted}

"""
import argparse
import csv
import os

import numpy as np

from custom_multiclassifier import CustomMulticlassifier

class_to_consider = [16, 23, 47, 49, 53, 67, 74, 81, 288, 343, 395, 396]

def predict_csv():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--configuration_models", help="Path of configuration models. Default is configuration.json", default=os.path.join(os.getcwd(), "configuration.json"))
    requiredNamed = parser.add_argument_group('Required named arguments')
    requiredNamed.add_argument("-r", "--results", help="Path of file in which write all the results. Mandatory!", required=True)
    requiredNamed.add_argument("-t", "--test_set", help="Path of directory with all csv files. Mandatory!", required=True)

    csv_path = parser.parse_args().test_set
    file_results = open(parser.parse_args().results, "w")
    csv_names = [os.path.join(csv_path, f) for f in os.listdir(csv_path)]
    configuration_path = parser.parse_args().configuration_models

    multiclassifier = CustomMulticlassifier(configuration_path)
    for csv_name in csv_names:
        features_int = np.empty((0, 128))
        if csv_name.split(".")[-1] == "csv":
            try:
                csv_reader = csv.reader(open(csv_name), delimiter=",")
                for row in csv_reader:
                    second_array = np.array([int(x) for x in row])
                    features_int = np.vstack((features_int, second_array))
                prediction = multiclassifier.predict(features_int)[0]
                file_results.write("{} {}\n".format(csv_name.split("/")[-1], prediction))
            except Exception as e:
                print(e)


if __name__ == "__main__":
    predict_csv()

