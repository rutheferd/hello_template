import shutil

import tensorflow as tf
import os
from tensorflow import keras
from matplotlib import pyplot as plt
import random as r
import model as m

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def change_input(input_file, batch_size):
    #check the different folders in the original input file name

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
        .batch(m.batch_size)
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
    img = tf.keras.utils.load_img(path_file, target_size = (height, width))
    img_array = tf.keras.utils.img_to_array(img)
    predictions = m.makePrediction(modelInput, img_array, train_ds.class_names)
    return predictions
    #print(img_array)

def gathering_data_confidence(train_ds, height, width, model):
    version = ["testing_set\\CatsDogs"]
    count = 0
    predict_colorsV2 = list()
    predict_colorsV3 = list()
    for directory_name in version: #colorsV# directory
        count += 1
        for testing_name in train_ds.class_names: #red, green, blue
            path = os.getcwd() + "\\" + directory_name + "\\" + testing_name
            files = os.listdir(path)
            for file_name in files: #files in red,green, or blue directory
                #print(file_name)
                prediction = m.makePrediction(model,path, train_ds.class_names)
                if count == 1:
                    # add the prediction for that file in the colorV2 list
                    predict_colorsV2.append(prediction)
                else:
                    # add the prediction for that file in the colorV3 list
                    predict_colorsV3.append(prediction)

def categorize(model, confidence_threshold, testing_directory_name, train_ds):
    try:
        os.mkdir(testing_directory_name + "\\" + "lessThan"
             + str(confidence_threshold * 100) + "% confident")
    except:
        print()

    for file in os.listdir(testing_directory_name):
        prediction, confidence = m.makePrediction(model, testing_directory_name + "\\"+ file, train_ds.class_names)
        if confidence < confidence_threshold:
            shutil.copyfile(file,  testing_directory_name + "\\" + "lessThan"
         + str(confidence_threshold * 100) + "% confident")


if __name__ == "__main__":
    batch_size = 10
    training, validation = change_input("ColorsV1", batch_size)
    test = "ColorsV3"
    width = 400
    height = 400
    model = m.createModel(width,height, len(training.class_names))
    m.trainModel(model, m.num_epochs ,training,validation)
    model.save("CDSavedModel")
    #pathF = os.getcwd() + "\\cat.4001.jpg"
    #img = tf.keras.utils.load_img(pathF, target_size=(height, width))
    #img_array = tf.keras.utils.img_to_array(img)
    #m.makePrediction(model, img_array ,training.class_names)


    #gathering_data_confidence(training, height, width, model)



