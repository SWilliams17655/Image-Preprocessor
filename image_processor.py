import cv2 as cv2
import numpy as np


def image_optical_flow_mask(image1, image2):
    """ Takes in two images then calculates the optical flow between the two masking anything not moving.
    :param image1: First image to be processed.
    :param image2: Second image to be processed.
    :return: An image with the non-moving regions masked
    """

    if len(image1.shape) == 3:  # If image is not already in greyscale converts it.
        image1_gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    else:
        image1_gray = image1

    if len(image2.shape) == 3:  # If image is not already in greyscale converts it.
        image2_gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    else:
        image2_gray = image2

    # Calculates dense optical flow by Farneback method
    flow = cv2.calcOpticalFlowFarneback(image1_gray, image2_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    # Computes the magnitude and angle of the 2D vectors
    magnitude, angle = cv2.cartToPolar(flow[:, :, 0], flow[:, :, 1], angleInDegrees=True)
    avg_mag = np.average(magnitude)
    std_mag = np.std(magnitude)
    image_out = np.where(np.logical_or((magnitude > (avg_mag + std_mag)), (magnitude < (avg_mag - std_mag))), image2, 0)

    return image_out


def image_optical_flow(image1, image2):
    """ Takes in two images then calculates the optical flow between the two creating a hsv representation of motion.
    :param image1: First image to be processed.
    :param image2: Second image to be processed.
    :return: An image with a hsv representation showing color for motion.
    """
    if len(image1.shape) == 3:  # If image is not already in greyscale converts it.
        image1_gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    else:
        image1_gray = image1

    if len(image2.shape) == 3:  # If image is not already in greyscale converts it.
        image2_gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    else:
        image2_gray = image2

    hsv_mask = np.zeros((image1_gray.shape[0], image1_gray.shape[1], 3), dtype="uint8")
    hsv_mask[:, :, 1] = 255

    # Calculates dense optical flow by Farneback method
    flow = cv2.calcOpticalFlowFarneback(image1_gray, image2_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    # Computes the magnitude and angle of the 2D vectors
    magnitude, angle = cv2.cartToPolar(flow[:, :, 0], flow[:, :, 1], angleInDegrees=True)

    hsv_mask[:, :, 0] = angle / 2

    hsv_mask[:, :, 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)

    # Converts HSV to RGB (BGR) color representation
    rgb = cv2.cvtColor(hsv_mask, cv2.COLOR_HSV2BGR)

    return rgb


def image_resize_dimensions(image, ratio=1.0):
    """ Opens an image as a BGR and rescales it if required.
    :param image: String representing file location to be opened.
    :param ratio: A floating point ratio representing a rescaling of the image.
    :return: An array representing the image opened.
    """

    scale_percent = ratio  # percent of original size
    image_width = int(image.shape[1] * scale_percent)
    image_height = int(image.shape[0] * scale_percent)
    dim = (image_width, image_height)

    return cv2.resize(image, dim, interpolation=cv2.INTER_AREA)  # resize image


def image_threshold(image, threshold=127, adaptive=False, otsu=False, kernel_size=3):
    """ A function that creates a threshold of the image being input [color or grey].
    :param image: The image to be input into the threshold.
    :param threshold: The threshold used for standard threshold and OTSU.
    :param otsu: Boolean representing if OTSU method should be used.
    :param adaptive: Boolean representing if adaptive or standard threshold should be used.
    :return: An image processed with the threshold algorithm.
    """
    if len(image.shape) == 3:  # If image is not already in greyscale converts it.
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if adaptive:
        return cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, kernel_size, 0)
    elif otsu:
        ret_val, image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        return image
    else:
        ret_val, image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY_INV)
        return image


def image_blur(image, kernel_size=5):
    """ Blurs the image using a gaussian blur to remove noise.
    :param image: Image to be blurred.
    :param kernel_size: Size of the kernel needs to be odd and will increase blurriness.
    :return: Blurred image.
    """
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)


def image_contour(threshold_image, original_image):
    """Takes a threshold image and the other an original image then returns a marked image
    and a list of contours. Recommend bluring and threshold be applied prior.
    :param threshold_image: Greyscale image with threshold and blur applied.
    :param original_image: Original image that will have the contours applied.
    :return: The original_image with contours applied, A list of contours
    """

    contours, hierarchy = cv2.findContours(threshold_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    external_contours = np.zeros(original_image.shape)
    cv2.drawContours(original_image, contours, contourIdx=-1, color=(0, 255, 0), thickness=1, lineType=cv2.LINE_AA)
    return original_image


def image_corner_detection(image, max_corners=60):
    """Takes an image as input and returns an image with points drawn and an array of corners.
    :param image: The image to be processed.
    :param max_corners: The maximum number of corners to be detected.
    :return: The image with editing complete, Array with all corners documented.
    """
    if len(image.shape) == 3:
        full_grey = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        full_grey = np.copy(image)

    corners = cv2.goodFeaturesToTrack(full_grey,
                                      maxCorners=max_corners,
                                      qualityLevel=0.01,
                                      minDistance=10,
                                      )
    corners = np.intp(corners)

    for i in corners:
        x, y = i.ravel()
        cv2.circle(image, (x, y), 5, (255, 0, 0), -1)

    return image, corners


def image_sobel_gradient(image):
    """ Calculates a standard image gradient which is used to detect edges within an image. A gradient is defined as
    a directional change in image intensity. This function uses Sobel in the X and Y direction then combines their
    results.
    :param image: Image the gradient will be applied too.
    :return: Image with the gradient.
    """
    if len(image.shape) == 3:
        full_grey = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        full_grey = np.copy(image)

    kernel_size = 3
    gradient_x = cv2.Sobel(full_grey, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=kernel_size)
    gradient_y = cv2.Sobel(full_grey, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=kernel_size)

    gradient_x = cv2.convertScaleAbs(gradient_x)  # Converting back to an 8-bit integer so OpenCV can operate
    gradient_y = cv2.convertScaleAbs(gradient_y)

    combined = cv2.addWeighted(gradient_x, 0.5, gradient_y, 0.5, 0)  # combines into a single image

    return combined


def image_canny_edges(image):
    """ Takes an image and highlights all edges which is a local extrema of the image gradient.
    :param image: Image where edges are to be detected.
    :return: image with the edges drawn
    """
    # processing to find the optimum lower and upper value
    med_val = np.median(image)
    lower = int(max(0, .7 * med_val))
    upper = int(min(255, 1.3 * med_val))
    edges = cv2.Canny(image=image, threshold1=lower, threshold2=upper)
    return edges


def image_kmean_segmentation(image, k):
    """Segments an image using K Mean algorithm.
    :param image: Image where segments are being detected.
    :param k: Number of segments to be detected.
    :return: image with the segments drawn as different colors.
    """
    # an array holding a list of different colors used based on size of K.
    color = np.array(
        [[0, 255, 0], [255, 0, 0], [0, 0, 255], [255, 255, 0], [0, 255, 255], [255, 0, 255], [119, 159, 68],
         [97, 220, 151], [228, 164, 166], [191, 75, 184], [229, 40, 22], [243, 177, 234], [96, 43, 149], [113, 39, 186],
         [131, 227, 133], [203, 15, 240], [181, 110, 167], [187, 63, 206], [202, 70, 10], [43, 146, 61], [185, 10, 209],
         [28, 79, 72], [75, 183, 187], [135, 125, 93], [253, 76, 44], [212, 9, 132], [126, 215, 56], [84, 198, 179],
         [115, 104, 183], [243, 188, 33], [29, 150, 16], [6, 224, 62], [150, 92, 249], [249, 106, 81], [15, 91, 39],
         [51, 210, 91], [110, 81, 133], [102, 155, 71], [135, 35, 102], [165, 157, 110], [121, 221, 60], [152, 193, 20],
         [163, 222, 237], [177, 97, 149], [55, 23, 226], [114, 54, 212], [68, 73, 88], [128, 53, 147], [214, 19, 144],
         [98, 165, 163], [53, 170, 70], [108, 15, 97], [5, 250, 78], [65, 6, 215], [152, 55, 172], [101, 198, 200],
         [87, 109, 216], [233, 240, 202], [46, 44, 128], [184, 247, 112], [75, 33, 136], [189, 143, 210],
         [123, 90, 167], [83, 35, 232], [182, 187, 68], [92, 199, 225], [182, 56, 22], [122, 223, 138], [233, 166, 43],
         [113, 81, 231], [245, 189, 2], [11, 127, 78], [118, 82, 157], [41, 47, 48], [113, 224, 107], [156, 7, 203],
         [25, 228, 33], [104, 141, 56], [74, 7, 244], [28, 85, 27], [45, 109, 211], [228, 255, 8], [23, 194, 114],
         [32, 225, 32], [25, 30, 126], [83, 163, 112], [137, 143, 65], [20, 52, 218], [167, 13, 230], [38, 0, 117],
         [70, 102, 249], [93, 20, 233], [31, 248, 67]])

    color = np.uint8(color)
    image_2d = image.reshape((-1, 3))  # Reshaping image to be 2D for K Mean processing.
    image_2d = np.float32(image_2d)  # Converting from 8 bit integer to float.

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)  # Stops when either criteria is met.

    # labels: An array [1-image size] with the designated label [1-K] for each pixel
    # center: An array [1-K] with the pixel color of that labeled section

    ret, label, center = cv2.kmeans(image_2d, k, None, criteria=criteria, attempts=10, flags=cv2.KMEANS_PP_CENTERS)

    segmented_data = color[label.flatten()]  # Uses np array technique to replace each image pixel with the color of
    # the center pixel. Example suppose we have a numpy array a= [1 2 3], so if we need to construct a numpy array
    # of [1 2 3 2 1] we can easily populate using the expression a [[0 1 2 1 0]].

    return segmented_data.reshape(image.shape)
