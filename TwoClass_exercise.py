import os
import cv2
import pandas
import numpy as np
from matplotlib import pyplot as plt


def mean_pix_intensity(input_path, img_list_class):
    """
    This function calculates the mean pixel intensity.

    :param input_path: The input path to the folder where the crops are located at
    :param img_list: a list of image files to be processed
    :return: returns the mean value of each classes's crops
    """
    img_mean_list = []

    for img in img_list_class:
        current_img_path = os.path.join(input_path, img)
        current_img = cv2.imread(current_img_path, -1)
        current_img_mean = np.round(np.mean(current_img), 2)
        img_mean_list.append([img, current_img_mean])

    # print('The images and their means are: ', img_mean_list, '\n')

    img_mean_list_class_values = [i[1] for i in img_mean_list]
    # print("img_mean_list_class_values", img_mean_list_class_values)
    return img_mean_list_class_values


def perimeter_differences(input_path, img_list_class):
    img_perimeter_list = []
    for img in img_list_class:
        current_img_path = os.path.join(input_path, img)
        current_img = cv2.imread(current_img_path)
        # imgray = cv2.cvtColor(current_img, cv2.COLOR_BGR2GRAY)
        # ret, thresh = cv2.threshold(imgray, 127, 255, 0)
        # contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # cnt = contours[0]

        edges = cv2.Canny(current_img, 100, 200)
        contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnt = contours[0]
        current_img_perimeter = np.round(cv2.arcLength(cnt, True), 2)
        img_perimeter_list.append([img, current_img_perimeter])

    # print('The images and their perimeters are: ', img_perimeter_list, '\n')
    img_perimeter_list_class_values = [i[1] for i in img_perimeter_list]
    return img_perimeter_list_class_values


def histogram_plotter(img_mean_list_class1_values, img_mean_list_class2_values):
    """

    :param img_mean_list_class1_values: The mean pixel intensity per image of each crop in class 1.
    :param img_mean_list_class2_values: The mean pixel intensity per image of each crop in class 2.
    :return: Histogram plot showing the two classes.
    """
    plt.title("134 crops each class")
    plt.hist(img_mean_list_class1_values, bins=10, range=(np.min(img_mean_list_class1_values),
                                                          np.max(img_mean_list_class1_values)),
             label='Neurites', alpha=0.7)
    plt.hist(img_mean_list_class2_values, bins=10, range=(np.min(img_mean_list_class2_values),
                                                          np.max(img_mean_list_class2_values)),
             label='Somas', alpha=0.7)
    plt.legend(loc='upper right')
    plt.show()

    return


def csv_file_updater(input_path, img_list, img_mean_list):
    csv_file_path = '\\'.join(str(input_path).split('\\')[0:-1])
    print("the csv file path: ", csv_file_path)
    csv_file = os.path.join(csv_file_path, 'classifier_output.csv')
    # print("the img_mean_list is: ", img_mean_list)
    img_mean_list_file_names = [i for i in img_list]
    img_mean_list_values = [j for j in img_mean_list]
    df = pandas.DataFrame(data={"Folder": input_path, "FileNames": img_mean_list_file_names,
                                "MeanPixIntensities": img_mean_list_values})
    df.to_csv(csv_file, sep=',', index=False)

    return


def main():
    img_mean_list_class_values = {}
    img_perimeter_list_class_values = {}

    for cl in class_list:
        # print("img_list is: ", img_list, '\n')
        assert os.path.exists(input_path[cl]), 'Please confirm the input path is correct.'
        img_list = os.listdir(input_path[cl])
        # img_list_class = [fn for fn in img_list if cl in fn]
        # print("img_list_class: ", img_list_class)
        img_mean_list_class_values[cl] = mean_pix_intensity(input_path[cl], img_list)
        csv_file_updater(input_path[cl], img_list, img_mean_list_class_values[cl])

        img_perimeter_list_class_values[cl] = perimeter_differences(input_path[cl], img_list)

    # TODO: csv file is not getting both classes, it overwrites
    print('The Class mean dictionary is: ', img_mean_list_class_values, '\n')
    print('The Class perimeter dictionary is: ', img_perimeter_list_class_values, '\n')

    histogram_plotter(img_mean_list_class_values[class_list[0]],
                      img_mean_list_class_values[class_list[1]])

    histogram_plotter(img_perimeter_list_class_values[class_list[0]],
                      img_perimeter_list_class_values[class_list[1]])


if __name__ == '__main__':
    input_path = \
        {'Neurites': 'C:\\Users\\sinad\\Dropbox (Gladstone)\\Feature_based_classification\\FIJI_SingleTp_N_CTR_1',
         'Somas': 'C:\\Users\\sinad\\Dropbox (Gladstone)\\Feature_based_classification\\FIJI_SingleTp_S_CTR_1'}
    # '/home/sinadabiri/Dropbox (Gladstone)/Feature_based_classification/ten_crops'

    class_list = ['Neurites', 'Somas']

    main()
