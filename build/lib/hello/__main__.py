import click

from hello.src import DataClass
from hello.src import model as m

from hello.src import there_command
from hello.src import DataHandler as DH
import os
from hello.src import training_command
from hello.src import predict_command

@click.group()
@click.version_option(package_name="image_classifier")
def main():
    """Image Classifier is a CLI tool that creates a machine learning model to classify images"""
    #model.
    pass

@main.command()
@click.option( "--model", "-m", type=click.STRING, help="If a model exists then use the model that will be tested.")
@click.option("--training","-tr",type=click.STRING,required=True,help="Adds the data that the model will be trained on.")
@click.option("--testing","-te",type=click.STRING,help = "Adds the data that the model will be tested on.")
@click.option("--epochs","-e",type=int,help = "Changes the number of epochs that will be done during training.")
@click.option("--batch","-b",type=int,help = "Changes the batch number that will be used during training.")
@click.option("--height","-h",type=int,help = "Changes the height of the images during training.")
@click.option("--confidence_threshold","-ct",type=int,help = "Changes the height of the images during training.")
@click.option("--width","-w",type=int,help = "Changes the width of the images during training.")
@click.option("--output","-o",type=click.STRING,help="Changes where the file will be created to store analysis about the model creation process.")
@click.option("--save_model", "-sm", type=click.STRING, help="Location of where the model will be saved if you want to save the model.")
def train(training, testing, batch, epochs, model, height, width, confidence_threshold, output, save_model):
    # Make a call to the model if it needs to be trained and saved somewhere
    print(epochs)
    epochsV = 8
    batchV = 32
    heightV = 400
    widthV = 400
    modelV = ""
    trainingV = ""
    testingV = ""
    ctV = -1
    outputV = "Output"
    saveV = ""

    print("Training", training)
    print("Testing", testing)


    if save_model:
        saveV = save_model
        print(save_model)
    if epochs:
        epochsV = epochs
        #print(epocs)
    if confidence_threshold:
        ctV = confidence_threshold
        #print(confidence_threshold)
    if batch:
        batchV = batch
        #print(batch)
    if height:
        heightV = height
        #print(heightV)
    if width:
        widthV = width
        #print(weightV)
    if output:
        outputV = output
    if model:
        # if mode is not in correct directory throw an exception
        if not os.path.isdir(model):
            raise Exception("Model is not in a directory.")
        elif not os.path.exists(model):
            raise Exception("This path does not exist.")
        modelV = model
        print(model)
    if training:
        # if mode is not in correct directory throw an exception
        if not os.path.isdir(training):
            raise Exception("Training data set is not in a directory.")
        elif not os.path.exists(training):
            raise Exception("This path does not exist.")
        trainingV = training
        print(training)
    if testing:
        # if mode is not in correct directory throw an exception
        if not os.path.isdir(testing):
            raise Exception("The testing data set is not in a directory.")
        elif not os.path.exists(testing):
            raise Exception("This path does not exist.")
        testingV = testing
        print(testing)
    print("ADS")
    #run.runTraining(epochsV, batchV, trainingV, testingV, heightV, widthV, modelV, ctV, outputV, saveV)
    training_command.run(epochsV, batchV, trainingV, testingV, heightV, widthV, modelV, ctV, outputV,saveV)
    print("Train")
    return

@main.command()
@click.option( "--model", "-m", type=click.STRING, required=True, help="If a model exists then use the model that will be tested.")
@click.option("--training","-tr",type=click.STRING,help="Adds the data that the model will be trained on.")
@click.option("--testing","-te",type=click.STRING, required=True, help = "Adds the data that the model will be tested on.")
@click.option("--epochs","-e",type=int,help = "Changes the number of epochs that will be done during training.")
@click.option("--batch","-b",type=int,help = "Changes the batch number that will be used during training.")
@click.option("--height","-h",type=int,help = "Changes the height of the images during training.")
@click.option("--confidence_threshold","-ct",type=int,help = "Changes the height of the images during training.")
@click.option("--width","-w",type=int,help = "Changes the width of the images during training.")
@click.option("--output","-o",type=click.STRING,help="Changes where the file will be created to store analysis about the model creation process.")
@click.option("--save_model", "-sm", type=click.STRING, help="Location of where the model will be saved if you want to save teh model.")

def predict(training, testing, batch, epochs, model, height, width, confidence_threshold, output,save_model):
    # Make a call to the model if it needs to be trained and saved somewhere
    print(epochs)
    epochsV = 8
    batchV = 32
    heightV = 400
    widthV = 400
    modelV = ""
    trainingV = ""
    testingV = ""
    ctV = -1
    outputV = "Output"
    saveV = ""

    print("Training", training)
    print("Testing", testing)

    if save_model:
        saveV = save_model
        print(save_model)
    if epochs:
        epochsV = epochs
        # print(epocs)
    if confidence_threshold:
        ctV = confidence_threshold
        # print(confidence_threshold)
    if batch:
        batchV = batch
        # print(batch)
    if height:
        heightV = height
        # print(heightV)
    if width:
        widthV = width
        # print(weightV)

    if output:
        if not os.path.isdir(output):
            raise Exception("Model is not in a directory.")
        elif not os.path.exists(output):
            raise Exception("This path does not exist.")
        outputV = output
        #print(output)

    if model:
        # if mode is not in correct directory throw an exception
        if not os.path.isdir(model):
            raise Exception("Model is not in a directory.")
        elif not os.path.exists(model):
            raise Exception("This path does not exist.")
        modelV = model
        #print(model)
    if training:
        # if mode is not in correct directory throw an exception
        if not os.path.isdir(training):
            raise Exception("Training data set is not in a directory.")
        elif not os.path.exists(training):
            raise Exception("This path does not exist.")
        trainingV = training
        #print(training)
    if testing:
        # if mode is not in correct directory throw an exception
        if not os.path.isdir(testing):
            raise Exception("The testing data set is not in a directory.")
        elif not os.path.exists(testing):
            raise Exception("This path does not exist.")
        testingV = testing
        #print(testing)

    #runTesting()
    predict_command.run(epochsV, batchV, trainingV, testingV, heightV, widthV, modelV, ctV, outputV,saveV)
    print("test")
    return


if __name__ == "__main__":
    main()
