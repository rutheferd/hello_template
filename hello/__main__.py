import click
from hello.src import there_command
from hello.src import model

@click.group()
@click.version_option(package_name="image_classifier")
def main():
    """Image Classifier is a CLI tool that creates a machine learning model to classify images"""
    #model.
    pass
"""
@click.option(
    "--model",
    "-m",
    required=True,
    type=str,
    is_flag=True,
    help = "Adds a model that will be tested.",
)
"""
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
    "--epocs",
    "-e",
    type=int,
    #is_flag=True,
    help = "Changes the number of epocs that will be done during training.",
)
@click.option(
    "--batch",
    "-b",
    type=int,
    #is_flag=True,
    help = "Changes the batch number that will be used during training.",
)
@main.command()
def model(training, testing, batch, epocs):
    # Make a call to the model if it needs to be trained and saved somewhere
    epocsV = 8
    batchV = 32
    print("Training", training)
    print("Testing", testing)
    if epocs:
        epocsV = epocs
        print(epocs)
    if batch:
        batchV = batch
        print(batch)

if __name__ == "__main__":
    main()
