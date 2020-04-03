import cv2
import numpy as np
import os
from matplotlib import pyplot as plt


def main():
    current_img_path = os.path.join(input_path, img_list[0])
    current_img = cv2.imread(current_img_path)
    plt.figure("img")
    plt.imshow(current_img)

    imgray = cv2.cvtColor(current_img, cv2.COLOR_BGR2GRAY)
    plt.figure("img gray scale")
    plt.imshow(imgray)

    ret, thresh = cv2.threshold(imgray, 127, 255, 0)
    plt.figure("img threshold")
    plt.imshow(thresh)

    edges = cv2.Canny(current_img, 100, 200)
    plt.figure("img Canny edge detection")
    plt.imshow(edges, cmap='gray')

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    plt.figure("img contours drawn")
    plt.imshow(cv2.drawContours(current_img, contours, -1, (0, 255, 0), 3))

    contours_mom, hierarchy_mom = cv2.findContours(thresh, 1, 2)
    # plt.imshow(cv2.drawContours(current_img, contours_mom, -1, (0,255,0), 3))
    cnt = contours_mom[0]
    print(contours_mom[0])
    M = cv2.moments(cnt)
    print(cnt)
    print(M)
    plt.figure("img contour moments drawn")
    plt.imshow(cv2.drawContours(current_img, cnt, -1, (0, 255, 0), thickness=3))

    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    print("centroid is x and y is: ", cx, cy)

    area = cv2.contourArea(cnt)
    print("The contour area is: ", area)

    perimeter = np.round(cv2.arcLength(cnt, True), 2)
    print("The contour perimeter is: ", perimeter)

    # Contour approximation:
    epsilon = 0.01 * perimeter
    approx = cv2.approxPolyDP(cnt, epsilon, True)
    plt.figure("img contour approximation")
    plt.imshow(cv2.drawContours(current_img, approx, -1, (0, 0, 255), thickness=1))


if __name__ == '__main__':
    input_path = 'C:\\Users\\sinad\\Dropbox (Gladstone)\\Feature_based_classification\\ten_crops2'
    # '/home/sinadabiri/Dropbox (Gladstone)/Feature_based_classification/ten_crops2'
    img_list = os.listdir(input_path)

    main()
