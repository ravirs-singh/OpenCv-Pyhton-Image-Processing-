from cv2 import cv2
import matplotlib.pyplot as plt
import numpy as np


image = cv2.imread("C:/Ravi/Python Open Cv/Seperating Colours/otsu.jpg",0)


bins_num = 256

# Get the image histogram
hist, bin_edges = np.histogram(image, bins=bins_num)

bin_mids = (bin_edges[:-1] + bin_edges[1:]) / 2.


weight1 = np.cumsum(hist)
weight2 = np.cumsum(hist[::-1])[::-1]

# Get the class means mu0(t)
mean1 = np.cumsum(hist * bin_mids) / weight1
# Get the class means mu1(t)
mean2 = (np.cumsum((hist * bin_mids)[::-1]) / weight2[::-1])[::-1]

inter_class_variance = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2


index_of_max_val = np.argmax(inter_class_variance)

threshold = bin_mids[:-1][index_of_max_val]
print("Otsu's algorithm implementation thresholding result: ", threshold)


otsu_threshold, image_result = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU,)

print("Obtained threshold: ", otsu_threshold)

cv2.imshow("image",image_result)

cv2.waitKey(0)
cv2.destroyAllWindows()
