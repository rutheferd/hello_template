import os
from hello.src import DataHandler as DH
from hello.src import DataClass
from hello.src import model as m
from tensorflow import keras
import keras.layers

data = DataClass.Parameters()
try:
    model = keras.models.load_model(data.model_file)
except:
    print()


def predict(numEpocs, numBatchSize, trainingPath, testingPath, height, width, modelPath, conf_thresh_val, output_loc, saveV):
    print("testing in runfile.")

    if output_loc == "Output":
        try:
            os.mkdir("Output")
            # logger.info("Make the Output directory")
        except:
            for f in os.listdir("Output"):
                os.remove(os.path.join("Output", f))
            # logger.info("Cleared Output directory")
        print(os.getcwd())
        # shutil.move(os.getcwd()+"logs.log", os.getcwd()+"Output")

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
        if trainingPath == "":
            raise Exception(
                "To test a model must be already made or testing data needs to inputted to train a new model.")
        else:
            training_d, validation = DH.change_input()
            if conf_thresh_val == -1:
                conf_thresh_val = 1 / len(training_d.class_names)
            d.num_confidence = conf_thresh_val
            m.createModel(len(training_d.class_names))
            m.trainModel(training_d, validation)
            DH.categorize(d.num_confidence, training_d.class_names)

    else:
        #training_d, validation = DH.change_input()
        numClasses = list()
        for i in os.listdir(testingPath):
            if os.path.isdir(i):
                numClasses.append(i)
        if conf_thresh_val == -1:
            conf_thresh_val = 1 / len(numClasses)
        d.num_confidence = conf_thresh_val
        m.model = keras.models.load_model(modelPath)
        #model.summary()
        #m.trainModel(training_d, validation)
        DH.categorize(d.num_confidence, numClasses)

    if not saveV == "":
        if output_loc == "Output":
            m.model.save(os.getcwd() + "\\Output")
        else:
            m.model.save(output_loc)
    return

def train(numEpocs, numBatchSize, trainingPath, testingPath, height, width, modelPath, conf_thresh_val, output_loc, saveV):
    print("training in runfile.")

    if output_loc == "Output":
        try:
            os.mkdir("Output")
            # logger.info("Make the Output directory")
        except:
            for f in os.listdir("Output"):
                os.remove(os.path.join("Output", f))
            # logger.info("Cleared Output directory")
        print(os.getcwd())
        # shutil.move(os.getcwd()+"logs.log", os.getcwd()+"Output")

        # set the data here
    d = DataClass.Parameters()
    # numEpocs, numBatchSize, height, width, trainingPath, testingPath, modelPath
    d.num_epochs = numEpocs
    d.batch_size = numBatchSize
    d.height_pixels = height
    d.width_pixels = width
    d.training_file = trainingPath
    d.test_file = testingPath
    d.model_file = modelPath
    d.output_location = output_loc

    if modelPath == "":
        training_d, validation = DH.change_input()
        if conf_thresh_val == -1:
            conf_thresh_val = 1 / len(training_d.class_names)
        d.num_confidence = conf_thresh_val
        m.createModel(len(training_d.class_names))
        m.trainModel(training_d, validation)
    else:
        training_d, validation = DH.change_input()
        if conf_thresh_val == -1:
            conf_thresh_val = 1 / len(training_d.class_names)
        d.num_confidence = conf_thresh_val
        m.model = keras.models.load_model(modelPath)
        model.summary()
        m.trainModel(training_d, validation)

    if not saveV == "":
        if output_loc == "Output":
            m.model.save(os.getcwd() + "\\Output")
        else:
            m.model.save(output_loc)
    return