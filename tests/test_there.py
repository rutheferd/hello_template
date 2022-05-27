import numpy
from click.testing import CliRunner
from hello.src import there_command
from hello.__main__ import main
import pytest
from hello.src import model as m
import tensorflow as tf
from keras.layers import MaxPooling2D
import tensorflow_datasets as tfds
from hello.src import DataHandler as DH
from hello.src import DataClass
from hello.src import model

data = DataClass.Parameters()

def test_there_function():
    name = "Austin"
    greeting = True
    assert (
        there_command.there(name, greeting)
        == "Hello there Austin, how are you?"
    )

    name = "Austin"
    assert there_command.there(name) == "Hello there Austin."

    name = "Austin"
    greeting = False
    assert there_command.there(name, greeting) == "Hello there Austin."

    pass

#
# def test_there_command():
#     # Testing with greeting
#     name = "Austin"
#     runner = CliRunner()
#     result = runner.invoke(main, ["there", "-n", name, "-g"])
#     assert result.output == "Hello there Austin, how are you?\n"
#
#     # Testing without greeting
#     name = "Obi-Wan"
#     result = runner.invoke(main, ["there", "-n", name])
#     assert result.output == "Hello there Obi-Wan.\n"
#
#     pass


def test_training():
    #makes a model for testing
    model = data.model

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Input(shape=(400, 400, 3)))
    model.add(tf.keras.layers.Conv2D(32, 3, padding = 'same' , activation='relu'))
    model.add(MaxPooling2D())
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(units=64, activation='relu'))
    model.add(tf.keras.layers.Dense(units = 2))

    model.compile(optimizer = 'adam',
                  loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True),
                  metrics = ['accuracy'])

    #save initial weights
    weights = model.layers[0].get_weights()

    #data prep
    DH.data.training_file = "training_set"
    DH.data.test_file = "testing_set"
    DH.data.num_epochs = 1
    train,val = DH.change_input()

    m.trainModel(train,val)

    #Save weights after training
    after_weights = data.model.layers[0].get_weights()

    #True if before weights and after weights are the same
    assert not numpy.array_equal(weights,after_weights,equal_nan=False )

def test_model_creation():
    model.createModel(2)
    #data.model initiall equals none but should not after model creation
    assert data.model != None

def test_data_saving():
    runner = CliRunner()
    result = runner.invoke(main," -m 1 train -tr 2 -t " )
    print(result.output)






