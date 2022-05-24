import shutil

import tensorflow as tf
import os
from tensorflow import keras
from matplotlib import pyplot as plt
import random as r

from hello.src import DataClass
from hello.src import model as m

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#dataclass can be changed in anoteher class
data  = DataClass.Parameters()

def change_input():
    #check the different folders in the original input file name
    input_file = data.training_file
    print("Data", data.training_file)
    print(input_file)
    batch_size = data.batch_size
    image_size = (400, 400)
    seedNum = r.randint(1,10000)
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        input_file,
        validation_split=0.2,
        subset="training",
        seed=seedNum,
        image_size=image_size,
        batch_size=batch_size,
    )
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        input_file,
        validation_split=0.2,
        subset="validation",
        seed=seedNum,
        image_size=image_size,
        batch_size=batch_size,
    )

    # class_names =  train_ds.class_names
    # plt.figure(figsize=(10, 10))
    # for images, labels in train_ds.take(1):
    #     for i in range(9):
    #         ax = plt.subplot(3, 3, i + 1)
    #         plt.imshow(images[i].numpy().astype("uint8"))
    #         plt.title(class_names[labels[i]])
    #         plt.axis("off")
    # plt.show()
    return train_ds, val_ds

def scale_resize(image, label):
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image,(224,224))
    return (image,label)
def scale_resize_dataset(dataset):
    ds = (
        dataset
        .map(scale_resize, num_parallel_calls = tf.data.experimental.AUTOTUNE)
        .batch(data.batch_size)
        .prefetch(tf.data.experimental.AUTOTUNE)
    )
    return ds

def testing(file_name, path, height, width,modelInput,train_ds):
    """width = 400
    height = 400
    file_name = "Red0.jpg"
    testing_name = "Red"
    directory_name = "ColorsV3"
    """
    #"C:\\Users\\sranjan31\\PycharmProjects\\gtriScreenDifferentiation\\ColorsV3\\Red\\"+
    #path = os.getcwd() + "\\" + directory_name + "\\" + testing_name + "\\" + file_name
    path_file = path + "\\" + file_name
    img = tf.keras.utils.load_img(path_file, target_size = (data.height_pixels, data.width_pixels))
    img_array = tf.keras.utils.img_to_array(img)
    predictions = m.makePrediction(modelInput, img_array, train_ds.class_names)
    return predictions
    #print(img_array)

def gathering_data_confidence(train_ds):
    height = data.height_pixels
    width = data.width_pixels
    model = data.model_file
    test_directory = data.test_file
    count = 0
    predict_colorsV2 = list()
    predict_colorsV3 = list()
    for directory_name in test_directory: #colorsV# directory
        count += 1
        for testing_name in train_ds.class_names: #red, green, blue
            path = test_directory + "\\" + testing_name
            files = os.listdir(path)
            for file_name in files: #files in red,green, or blue directory
                #print(file_name)
                prediction = m.makePrediction(path, train_ds.class_names)
                if count == 1:
                    # add the prediction for that file in the colorV2 list
                    predict_colorsV2.append(prediction)
                else:
                    # add the prediction for that file in the colorV3 list
                    predict_colorsV3.append(prediction)

def categorize(confidence_threshold, train_ds):
    testing_directory_name = data.test_file
    try:
        os.mkdir("lessThan"
             + str(confidence_threshold * 100) + "% confident")
    except:
        for f in os.listdir("lessThan"
             + str(confidence_threshold * 100) + "% confident"):
            os.remove(os.path.join("lessThan"
             + str(confidence_threshold * 100) + "% confident",f))
    try:
        os.mkdir("moreThan"
             + str(confidence_threshold * 100) + "% confident")
    except:
        for f in os.listdir("moreThan"
             + str(confidence_threshold * 100) + "% confident"):
            os.remove(os.path.join("moreThan"
             + str(confidence_threshold * 100) + "% confident",f))
    for file in os.listdir(testing_directory_name):
        img = tf.keras.utils.load_img(testing_directory_name + "\\" + file, target_size=(data.width_pixels, data.height_pixels))
        img_array = tf.keras.utils.img_to_array(img)
        prediction, confidence = m.makePrediction( img_array, train_ds.class_names)
        if confidence < confidence_threshold * 100:
            shutil.copyfile(testing_directory_name + "\\" + file,    "lessThan"
         + str(confidence_threshold * 100) + "% confident\\" + str(round(confidence,2)) + "Prediction;" + prediction + "Actual;" + file)
        else:
            shutil.copyfile(testing_directory_name + "\\" + file, "moreThan"
                            + str(confidence_threshold * 100) + "% confident\\" + str(
                round(confidence, 2)) + "Prediction;" + prediction + "Actual;" + file)


if __name__ == "__main__":
    training, validation = change_input()
    test = "ColorsV3"

    model = m.createModel(len(training.class_names))
    print("DMNKLSNKLDANKASNKL:"+ str(len(training.class_names)))
    m.trainModel(training,validation)
    model.save("CDSavedModel")
    #pathF = os.getcwd() + "\\cat.4001.jpg"
    #img = tf.keras.utils.load_img(pathF, target_size=(height, width))
    #img_array = tf.keras.utils.img_to_array(img)
    #m.makePrediction(model, img_array ,training.class_names)


    #gathering_data_confidence(training, height, width, model)



