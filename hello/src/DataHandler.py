import shutil

import tensorflow as tf
import os
from tensorflow import keras
from matplotlib import pyplot as plt
import random as r

from hello.src import DataClass
from hello.src import model as m
from hello.src.logger import logger

from mdutils.mdutils import MdUtils
from mdutils import Html

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#dataclass can be changed in anoteher class
data  = DataClass.Parameters()
def change_input():
    #check the different folders in the original input file name
    logger.info("Converting directory into training and validation datasets")
    input_file = data.training_file
    print(input_file)
    batch_size = data.batch_size
    image_size = (data.height_pixels, data.width_pixels)
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
    logger.info("Making predictions on V2 and V3 colors")
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
    logger.info("Making directories using " + str(confidence_threshold * 100) + " threshold")
    try:
        os.mkdir("lessThan"
             + str(confidence_threshold * 100) + "% confident")
        logger.info("Made less than confidence threshold directory")
    except:
        for f in os.listdir("lessThan"
             + str(confidence_threshold * 100) + "% confident"):
            os.remove(os.path.join("lessThan"
             + str(confidence_threshold * 100) + "% confident",f))
        logger.info("Cleared less than confidence threshold directory")
    try:
        os.mkdir("moreThan"
             + str(confidence_threshold * 100) + "% confident")
        logger.info("Made more than confidence threshold directory")
    except:
        for f in os.listdir("moreThan"
             + str(confidence_threshold * 100) + "% confident"):
            os.remove(os.path.join("moreThan"
             + str(confidence_threshold * 100) + "% confident",f))
        logger.info("Cleared more than confidence threshold directory")
    logger.info("Making predictions on test dataset and organizing entries into confidence directories")
    mdFile = MdUtils(file_name="Confidence and Accuracy Report", title="Confidence and Accuracy Report")
    above_threshold = list()
    below_threshold = list()
    for sub in os.listdir(testing_directory_name):
        for file in sub:
            img = tf.keras.utils.load_img(testing_directory_name + "\\" + file, target_size=(data.width_pixels, data.height_pixels))
            img_array = tf.keras.utils.img_to_array(img)
            prediction, confidence = m.makePrediction( img_array, train_ds.class_names)
            """if confidence < confidence_threshold * 100:
                shutil.copyfile(testing_directory_name + "\\" + file,    "lessThan"
             + str(confidence_threshold * 100) + "% confident\\" + str(round(confidence,2)) + "Prediction;" + prediction + "Actual;" + file)
            else:
                shutil.copyfile(testing_directory_name + "\\" + file, "moreThan"
                                + str(confidence_threshold * 100) + "% confident\\" + str(
                    round(confidence, 2)) + "Prediction;" + prediction + "Actual;" + file)"""
            if confidence < confidence_threshold*100:
                # add the values to the arrays
                below_threshold.append((confidence, prediction, sub, testing_directory_name + "\\" + sub + "\\" +file))
            else:
                above_threshold.append((confidence, prediction, sub, testing_directory_name + "\\" + sub + "\\" + file))
    #print("AT", above_threshold)
    above_avg_accuracy, above_avg_confidence = caluclate_average(above_threshold)
    below_avg_accuracy, below_avg_confidence = caluclate_average(below_threshold)
    mdFile.new_header(level=1, title="Confidence/Accuracy Above the Threshold")
    mdFile.new_paragraph("The average confidence level above the threshold is: " + str(above_avg_confidence))
    mdFile.new_paragraph("The average accuracy level above the threshold is: " + str(above_avg_accuracy))
    mdFile.new_paragraph("The data is in Appendix A")

    mdFile.new_header(level=2, title="Confidence/Accuracy Below the Threshold")
    mdFile.new_paragraph("The average confidence level below the threshold is: " + str(below_avg_confidence))
    mdFile.new_paragraph("The average accuracy level below the threshold is: " + str(below_avg_accuracy))
    mdFile.new_paragraph("The data is in Appendix B")

    mdFile.new_header(level=3, title="Appendix A")
    for i in above_threshold:
        mdFile.write("Path to Image: "+ str(i[3])+"\t"+"Confidence Level: " + str(i[0])+"\t"+"Predicted Label: "+str(i[1])+"\t"+"Actual Label: "+str(i[2])+"\n")

    mdFile.new_header(level=4, title="Appendix B")
    for i in below_threshold:
        mdFile.write("Path to Image: "+ str(i[3])+"\t"+"Confidence Level: " + str(i[0])+"\t"+"Predicted Label: "+str(i[1])+"\t"+"Actual Label: "+str(i[2])+"\n")
    mdFile.create_md_file()
    logger.info("Finished predicting test data ")

def caluclate_average(threshold_list):
    avg_confidence = 0
    avg_accuracy = 0
    counter = 0
    #print(threshold_list)
    for i in threshold_list:
        counter += 1
        avg_confidence += i[0]
        if i[1] == i[2]:
            avg_accuracy += 1
    if counter == 0:
        return 0, 0
    return avg_accuracy/counter, avg_confidence/counter



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



