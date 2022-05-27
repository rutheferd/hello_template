import logging
import os
from hello.src import DataHandler as DH
from hello.src import DataClass
from hello.src import model as m
from tensorflow import keras
import keras.layers
import shutil

data = DataClass.Parameters()
try:
    model = keras.models.load_model(data.model_file)
except:
    print()

def run(epochsV, batchV, trainingV, testingV, heightV, widthV, modelV, ctV, outputV,saveV):

    predict(epochsV, batchV, trainingV, testingV, heightV, widthV, modelV, ctV, outputV,saveV)
    logging.basicConfig(
        format="[%(asctime)s] %(levelname)s: %(message)s",
        datefmt="%m/%d/%Y %I:%M:%S %p",
        filename=outputV+"/logs.log",
        level=logging.INFO
    )
    LOGGER = logging.getLogger()
    LOGGER.info('Master runs')
    return

def predict(numEpocs, numBatchSize, trainingPath, testingPath, height, width, modelPath, conf_thresh_val, output_loc, saveV):
    print("testing in runfile.")

    if output_loc == "Output":
        print("Bleh")
    # set the data here
    d = DataClass.Parameters()
    d.num_epochs = numEpocs
    d.batch_size = numBatchSize
    d.height_pixels = height
    d.width_pixels = width
    d.training_file = trainingPath
    d.test_file = testingPath
    d.model_file = modelPath
    d.output_location = output_loc
    if modelPath == "":
        raise Exception("You must have a model to predict values.")

    else:
        #training_d, validation = DH.change_input()
        numClasses = list()
        for i in os.listdir(testingPath):
            #Fixed bug here where num classes would be empty due to i alone not being a valid directory
            if os.path.isdir(testingPath + "/" + i):
                numClasses.append(i)
        print(numClasses)
        if conf_thresh_val == -1:
            conf_thresh_val = 1 / len(numClasses)
        d.num_confidence = conf_thresh_val
        m.model = keras.models.load_model(modelPath)
        #model.summary()
        #m.trainModel(training_d, validation)
        DH.categorize(d.num_confidence, numClasses)

    if output_loc == "Output":
        m.model.save(os.getcwd() + "/Output/Model")
    else:
        m.model.save(output_loc)
    return
