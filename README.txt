To use this project you need theese two additional libraries

sounddevice==0.3.12
SoundFile==0.10.2


Theese are the principal scripts:

live.py is the script to run the live. Takes two not mandatory argument. -w for the size of windows and -c to choose the configuration file

training.py is the script to run the training of the model.

record_and_save.py is the script to define the thread that record and save the audio recorded in a .wav file during the live.

read_tfrecords.py is the script that have functions to extract data from the dataset and tfrecords file.

prediction_csv.py is the script to run the prediction of all csv files of the test set.
            It takes two mandatory argument that are -r result file path and -t for the test set directory
            and one argument not mandatory that is the configuration file path.

extract_audioset_embedding.py is the script that use all others file and the vggish_model to extract features from the audio file.
