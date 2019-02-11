import json
from keras.models import load_model
import numpy as np

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

        for models in models_configuration[MODELS_FIELD]:
            self.models.append(load_model(models[PATH_FIELD]))
            if models[STEPS_FIELD] in self.steps:
                print("There is already a model with this number of steps: {}".format(models[STEPS_FIELD]))
                raise Exception()
            self.steps.append(models[STEPS_FIELD])
            if models[STEPS_FIELD] > self.max_steps:
                self.max_steps = models[STEPS_FIELD]

        if EXPERT_MODEL in models_configuration:
            self.expert_model = load_model(models_configuration[EXPERT_MODEL][PATH_FIELD])
            self.expert_steps = models_configuration[EXPERT_MODEL][STEPS_FIELD]
            self.expert_classes = models_configuration[EXPERT_MODEL][CLASSES_FIELD]

    def prediction_different_windows(self, array, model, steps_model):
        steps_array = array.shape[0]
        if steps_model == steps_array:
            predict = model.predict(np.array([array]))[0]
            self.prediction = np.argmax(predict)
            self.accuracy = predict[self.prediction]
        else:
            predictions = []
            accuracy = []
            for i in range(steps_array - steps_model + 1):
                predict = model.predict(np.array([array[i:i+steps_model]]))[0]
                index = np.argmax(predict)
                predictions.append(index)
                accuracy.append(predict[index])

            index = np.argmax(np.array(predictions))
            self.prediction = predictions[index]
            self.accuracy = accuracy[index]

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

        if self.expert_model is not None and self.prediction in self.expert_classes:
            self.prediction_different_windows(array, self.expert_model, self.expert_steps)

        return self.classes_id[self.prediction], self.accuracy








