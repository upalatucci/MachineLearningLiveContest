'''
    CustomMulticlassifier is used in the project to assembly a multiclassifier with different models.

    The configuration given in input to the constructor should have and array of models and for each model contain
    information concerns number of steps supported from the model and the path to use for load the model.

    The CustomMulticlassifier choose the best model for the given audio array to predict.
    The algorithm choose the model with the possible higher steps and use it for the prediction.
    If the audio array have a major dimension in steps of the model's steps, the model give different
    predictions. The class with the higher probability in theese predictions is returned with the average probability.
'''

import json

import numpy as np
from keras.models import load_model

MODELS_FIELD = "models"
PATH_FIELD = "path"
STEPS_FIELD = "steps"
EXPERT_MODEL = "expert_model"
CLASSES_FIELD = "classes"


class CustomMulticlassifier:
    def __init__(self, configuration):
        models_configuration = json.loads(open(configuration, "r").read())
        self.models = []
        self.steps = []
        self.expert_model = None
        self.classes_id = [16, 23, 47, 49, 53, 67, 74, 81, 288, 343, 395, 396]
        self.max_steps = 0
        self.min_steps = None

        try:
            for models in models_configuration[MODELS_FIELD]:
                self.models.append(load_model(models[PATH_FIELD]))
                if models[STEPS_FIELD] in self.steps:
                    print("There is already a model with this number of steps: {}".format(models[STEPS_FIELD]))
                    raise Exception()
                self.steps.append(models[STEPS_FIELD])
                if models[STEPS_FIELD] > self.max_steps:
                    self.max_steps = models[STEPS_FIELD]
                if self.min_steps is None or models[STEPS_FIELD] < self.min_steps:
                    self.min_steps = models[STEPS_FIELD]

        except:
            print("You should insert into the configuration file for every model the path and steps")
            raise Exception()

        if len(self.models) == 0:
            print("You should insert at least one model into the configuration file with path and steps")
            raise Exception()

    def prediction_different_windows(self, array, model, steps_model):
        steps_array = array.shape[0]
        if steps_model == steps_array:
            predict = model.predict(np.array([array]))[0]
            self.prediction = np.argmax(predict)
            self.accuracy = predict[self.prediction]
        else:
            predictions = np.zeros((len(self.classes_id)))
            for i in range(steps_array - steps_model + 1):
                predict = model.predict(np.array([array[i:i+steps_model]]))[0]
                predictions = np.sum([predictions, predict], axis=0)

            self.prediction = np.argmax(np.array(predictions))
            self.accuracy = predictions[self.prediction]/(steps_array - steps_model + 1)

    def predict(self, array):
        steps_array = array.shape[0]
        steps_model_chose = None
        model_to_chose = None

        for i in range(len(self.steps)):
            if steps_array == self.steps[i]:
                model_to_chose = self.models[i]
                steps_model_chose = self.steps[i]
            else:
                try:
                    if self.steps[i-1] < steps_array < self.steps[i]:
                        model_to_chose = self.models[i-1]
                        steps_model_chose = self.steps[i-1]
                except:
                    pass

        if model_to_chose is None:
            model_to_chose = self.models[-1]
            steps_model_chose = self.steps[-1]

        self.prediction_different_windows(array, model_to_chose, steps_model_chose)

        return self.classes_id[self.prediction], self.accuracy








