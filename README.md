This is an image processing tool which includes a library of common image preprocessing functions I use. This includes edge detection, segmentation, corner detection, and optical flow to name a few. 

Purpose: This library is intended for image processing prior to use by a higher-level function. What each library returns is a processed image allowing the user to create a pre-processing pipe prior to a computer vision algorithm.

How: Functions take in a one or three channel image of any size then the user can call each processing function in order. What is returned is a processed image to be fed into the next algorithm.

Functions:
image_optical_flow_mask(image1, image2): Takes in two images then calculates the optical flow between the two masking anything not moving.
def image_optical_flow(image1, image2): Takes in two images then calculates the optical flow between the two creating a hsv representation of motion.

image_resize_dimensions(image, ratio=1.0): Opens an image as a BGR and rescales it if required.

image_threshold(image, threshold=127, adaptive=False, otsu=False, kernel_size=3): A function that creates a threshold of the image being input [color or grey].

image_blur(image, kernel_size=5): Blurs the image using a gaussian blur to remove noise.

image_contour(threshold_image, original_image): Takes a threshold image and the other an original image then returns a marked image
    and a list of contours. Recommend bluring and threshold be applied prior.

image_corner_detection(image, max_corners=60): Takes an image as input and returns an image with points drawn and an array of corners.

image_sobel_gradient(image): Calculates a standard image gradient which is used to detect edges within an image. A gradient is defined as a directional change in 
image intensity. This function uses Sobel in the X and Y direction then combines their results.

image_canny_edges(image): Takes an image and highlights all edges which is a local extrema of the image gradient.

image_kmean_segmentation(image, k): Segments an image using K Mean algorithm.
