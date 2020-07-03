import os
import cv2
import math
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


def fft_hpf_differences(input_path, img_list_class):

    img_fft_hpf_list = []
    for img in img_list_class:
        current_img_path = os.path.join(input_path, img)
        current_img = cv2.imread(current_img_path, 0)
        row, col = current_img.shape
        centerRow, centerCol = int(row/2), int(col/2)
        centerRectangle = 7
        dft = cv2.dft(np.float32(current_img), flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)
        # High pass filtering: blocking a square in the middle
        dft_shift[centerRow - centerRectangle: centerRow + centerRectangle,
                     centerCol - centerRectangle: centerCol + centerRectangle] = 0
        # Going back: inverse fft
        dft_ifft = np.fft.ifftshift(dft_shift)
        img_back = cv2.idft(dft_ifft)
        img_back = 20*np.log(cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1]))
        median_img_back = np.median(img_back.ravel())
        img_fft_hpf_list.append([img, median_img_back])

    # print('The images and their median pix after fft hpf are: ', img_fft_hpf_list, '\n')
    img_fft_hpf_list_class_values = [i[1] for i in img_fft_hpf_list]

    return img_fft_hpf_list_class_values


def fft_power_differences(input_path, img_list_class):

    img_fft_power_list = []
    for img in img_list_class:
        current_img_path = os.path.join(input_path, img)
        current_img = cv2.imread(current_img_path, 0)
        row, col = current_img.shape
        centerRow, centerCol = int(row/2), int(col/2)
        centerRectangle = 7
        dft = cv2.dft(np.float32(current_img), flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)
        # imgMagnitude = 20*np.log(cv2.magnitude(dft_shift[:,:,0], dft_shift[:,:,1]))
        # power = imgMagnitude**2
        # median_pow = np.median(power.ravel())

        # High pass filtering: blocking a square in the middle
        dft_shift[centerRow - centerRectangle: centerRow + centerRectangle,
        centerCol - centerRectangle: centerCol + centerRectangle] = 0
        # Going back: inverse fft
        dft_ifft = np.fft.ifftshift(dft_shift)

        img_back = cv2.idft(dft_ifft)
        img_back = 20*np.log((cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1]))**2)
        median_img_back = np.median(img_back.ravel())
        img_fft_power_list.append([img, median_img_back])

    # print('The images and their median pix after fft hpf power are: ', img_fft_power_list, '\n')
    img_fft_power_list_class_values = [i[1] for i in img_fft_power_list]

    return img_fft_power_list_class_values


def eccentricity_differences(input_path, img_list_class):
    img_eccentricity_list = []
    for img in img_list_class:
        current_img_path = os.path.join(input_path, img)
        current_img = cv2.imread(current_img_path)
        current_img_normalized = (current_img - current_img.min()) / (current_img.max() - current_img.min())
        current_img_normalized_mean = current_img_normalized.mean()
        current_img_normalized_std = current_img_normalized.std()
        mean_correction_factor = 0.25 - current_img_normalized_mean
        stdDev_correction_factor = 0.125 / current_img_normalized_std
        current_img_standardized = (current_img_normalized - mean_correction_factor) * stdDev_correction_factor
        current_img_standardized_corrected = current_img_standardized + (0.25 - current_img_standardized.mean())
        current_img_standardized_clipped = current_img_standardized_corrected.clip(0.02, 0.9)
        current_img_standardized_clipped_rescaled = np.array(current_img_standardized_clipped * pow(2,8), dtype=np.uint8)

        imgray = cv2.cvtColor(current_img_standardized_clipped_rescaled, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(imgray, 150, 255, 0)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contour_length = []
        for c in contours:
            contour_length.append(len(c))

        # print("the length of contours are: ", contour_length)
        contour_max_idx = contour_length.index(max(contour_length))
        # print("the index of the max contour is: ", contour_max_idx)
        cnt = contours[contour_max_idx]
        # cnt = contours[0]
        # print("the contours points are: ", contours)
        threshold_ellipse = cv2.fitEllipse(cnt)
        (center, axis, orientation) = threshold_ellipse
        major_axis = max(axis)
        minor_axis = min(axis)
        # print("the major and minor axis are: ", major_axis, minor_axis)
        eccentricity = np.sqrt(1-(minor_axis/major_axis)**2)
        current_img_eccentricity = np.round(eccentricity, 2)
        img_eccentricity_list.append([img, current_img_eccentricity])

    print('The images and their Eccentricities are: ', img_eccentricity_list, '\n')
    img_eccentricity_list_class_values = [i[1] for i in img_eccentricity_list]
    return img_eccentricity_list_class_values


def mean_circle_pix_intensity(input_path, img_list_class):
    img_mean_list = []
    blank_img = np.zeros((300, 300, 3), dtype=np.uint8)
    circle_mask = cv2.circle(blank_img, (150, 150), 25, (1, 1, 1), -1)

    for img in img_list_class:
        current_img_path = os.path.join(input_path, img)
        current_img = cv2.imread(current_img_path)
        current_img_filtered = current_img*circle_mask
        current_img_filtered_mean = np.round(np.mean(current_img_filtered), 2)
        img_mean_list.append(current_img_filtered_mean)
    return img_mean_list


def histogram_plotter(img_mean_list_class1_values, img_mean_list_class2_values, title):
    """

    :param title:
    :param img_mean_list_class1_values: The mean pixel intensity per image of each crop in class 1.
    :param img_mean_list_class2_values: The mean pixel intensity per image of each crop in class 2.
    :return: Histogram plot showing the two classes.
    """
    plt.figure(title)
    plt.title("134 crops each class")
    # bins_class1 = np.linspace(math.ceil(min(img_mean_list_class1_values)),
    #                           math.floor(max(img_mean_list_class1_values)),
    #                           10)
    plt.hist(img_mean_list_class1_values, bins=30, range=(np.min(img_mean_list_class1_values),
                                                                   np.max(img_mean_list_class1_values)),
             label='Class1', alpha=0.7)
    # bins_class2 = np.linspace(math.ceil(min(img_mean_list_class2_values)),
    #                           math.floor(max(img_mean_list_class2_values)),
    #                           10)
    plt.hist(img_mean_list_class2_values, bins=30, range=(np.min(img_mean_list_class2_values),
                                                                   np.max(img_mean_list_class2_values)),
             label='Class2', alpha=0.7)
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
    img_fft_hpf_list_class_values = {}
    img_fft_power_list_class_values = {}
    img_eccentricity_list_class_values = {}
    img_circle_mean_list_class_values = {}

    for cl in class_list:
        # print("img_list is: ", img_list, '\n')
        assert os.path.exists(input_path[cl]), 'Please confirm the input path is correct.'
        img_list = os.listdir(input_path[cl])
        # img_list_class = [fn for fn in img_list if cl in fn]
        # print("img_list_class: ", img_list_class)
        img_mean_list_class_values[cl] = mean_pix_intensity(input_path[cl], img_list)
        # csv_file_updater(input_path[cl], img_list, img_mean_list_class_values[cl])
        img_perimeter_list_class_values[cl] = perimeter_differences(input_path[cl], img_list)

        img_fft_hpf_list_class_values[cl] = fft_hpf_differences(input_path[cl], img_list)
        img_fft_power_list_class_values[cl] = fft_power_differences(input_path[cl], img_list)

        img_eccentricity_list_class_values[cl] = eccentricity_differences(input_path[cl], img_list)
        img_circle_mean_list_class_values[cl] = mean_circle_pix_intensity(input_path[cl], img_list)



    # TODO: csv file is not getting both classes, it overwrites
    # print('The Class mean dictionary is: ', img_mean_list_class_values, '\n')
    # print('The Class perimeter dictionary is: ', img_perimeter_list_class_values, '\n')
    # print('The Class fft hpf dictionary is: ', img_fft_hpf_list_class_values, '\n')
    histogram_plotter(img_mean_list_class_values[class_list[0]],
                      img_mean_list_class_values[class_list[1]], 'mean pix differences')

    histogram_plotter(img_perimeter_list_class_values[class_list[0]],
                      img_perimeter_list_class_values[class_list[1]], 'perimeter differences')

    # histogram_plotter(img_fft_hpf_list_class_values[class_list[0]],
    #                   img_fft_hpf_list_class_values[class_list[1]], 'fft hpf differences')
    #
    # histogram_plotter(img_fft_power_list_class_values[class_list[0]],
    #                   img_fft_power_list_class_values[class_list[1]], 'fft hpf power differences')

    histogram_plotter(img_eccentricity_list_class_values[class_list[0]],
                      img_eccentricity_list_class_values[class_list[1]], 'Eccentricity differences')
    histogram_plotter(img_circle_mean_list_class_values[class_list[0]],
                      img_circle_mean_list_class_values[class_list[1]], 'mean circle pix differences')


if __name__ == '__main__':
    input_path = \
        {'Class1': 'C:\\Users\\sinad\\Dropbox (Gladstone)\\Feature_based_classification\\FIJI_SingleTp_S_CTR_1',
         'Class2': 'C:\\Users\\sinad\\Dropbox (Gladstone)\\Feature_based_classification\\FIJI_SingleTp_N_CTR_1'}
    # '/home/sinadabiri/Dropbox (Gladstone)/Feature_based_classification/ten_crops'

    class_list = ['Class1', 'Class2']

    main()
