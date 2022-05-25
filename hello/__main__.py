import click

from hello.src import DataClass
from hello.src import model as m

from hello.src import there_command
from hello.src import DataHandler as DH
import os

@click.group()
@click.version_option(package_name="image_classifier")
def main():
    """Image Classifier is a CLI tool that creates a machine learning model to classify images"""
    #model.
    pass

@click.option(
    "--model",
    "-m",
    type=click.STRING,
    help="If a model exists then use the model that will be tested.",
)

@click.option(
    "--training",
    "-tr",
    type=click.STRING,
    required=True,
    #is_flag=True,
    help="Adds the data that the model will be trained on.",
)
@click.option(
    "--testing",
    "-te",
    type=click.STRING,
    required=True,
    #is_flag=True,
    help = "Adds the data that the model will be tested on.",
)
@click.option(
    "--epochs",
    "-e",
    type=int,
    #is_flag=True,
    help = "Changes the number of epochs that will be done during training.",
)
@click.option(
    "--batch",
    "-b",
    type=int,
    #is_flag=True,
    help = "Changes the batch number that will be used during training.",
)
@click.option(
    "--height",
    "-h",
    type=int,
    #is_flag=True,
    help = "Changes the height of the images during training.",
)
@click.option(
    "--confidence_threshold",
    "-ct",
    type=int,
    #is_flag=True,
    help = "Changes the height of the images during training.",
)
@click.option(
    "--width",
    "-w",
    type=int,
    #is_flag=True,
    help = "Changes the width of the images during training.",
)
@main.command()

def model(training, testing, batch, epocs, model, height, width, confidence_threshold):
    # Make a call to the model if it needs to be trained and saved somewhere
    epochsV = 8
    batchV = 32
    heightV = 400
    widthV = 400
    modelV = ""
    trainingV = ""
    testingV = ""
    ctV = -1

    '''
    print("Training", training)
    print("Testing", testing)
    '''


    if epocs:
        epocsV = epocs
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

    m.runNewModel(epocsV, batchV, trainingV, testingV, heightV, widthV, modelV, ctV)

    return



if __name__ == "__main__":
    main()
