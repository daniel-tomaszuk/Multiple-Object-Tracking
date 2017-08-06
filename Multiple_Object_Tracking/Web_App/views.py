from django.shortcuts import render
import numpy as np
import cv2
import sys

import matplotlib.pyplot as plt
from filterpy.gh import GHFilter
from munkres import Munkres, print_matrix


def otsu_binary(img):
    """
    Otsu binarization function.
    :param img: Image to binarize - should be in greyscale.
    :return: Image after binarization.
    """
    # check if input image is in grayscale (2D)
    try:
        if img.shape[2]:
            # if there is 3rd dimension
            sys.exit('otsu_binary(img) input image should be in grayscale!')
    except IndexError:
        pass  # image doesn't have 3rd dimension - proceed

    # plt.close('all')
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    # find normalized_histogram, and its cumulative distribution function
    hist = cv2.calcHist([blur], [0], None, [256], [0, 256])
    hist_norm = hist.ravel() / hist.max()
    Q = hist_norm.cumsum()
    bins = np.arange(256)
    fn_min = np.inf
    thresh = -1

    for i in range(1, 255):
        p1, p2 = np.hsplit(hist_norm, [i])  # probabilities
        q1 = Q[i]
        q2 = Q[255] - q1  # cum sum of classes
        b1, b2 = np.hsplit(bins, [i])  # weights
        # finding means and variances
        m1 = np.sum(p1 * b1) / q1
        m2 = np.sum(p2 * b2) / q2
        v1, v2 = np.sum(((b1 - m1) ** 2) * p1) / q1, np.sum(
            ((b2 - m2) ** 2) * p2) / q2
        # calculates the minimization function
        fn = v1 * q1 + v2 * q2
        if fn < fn_min:
            fn_min = fn
            thresh = i
    # find otsu's threshold value with OpenCV function
    ret, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY +
                              cv2.THRESH_OTSU)
    # print("{} {}".format(thresh, ret))

    ret, img_thresh1 = cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY)
    return img_thresh1


def select_frames(video, frame_start, frame_stop):
    """
    Function that return selected frames from video.
    :param video: string, video whom frames are selected,
    :param frame_start: integer, frame from selection should be started
    :param frame_stop: integer, ending frame of selected section
    :return: video fragment <start_frame, stop_frame>
    """
    cap = cv2.VideoCapture(video)
    # font = cv2.FONT_HERSHEY_SIMPLEX
    video_fragment = []
    cap.set(1, frame_start)

    while cap.isOpened():
        ret, frame = cap.read()
        video_fragment.append(frame)
        if cv2.waitKey(1) & 0xFF == ord('q') or not ret \
                or (cap.get(1)) == frame_stop + 1:
            break
    #     # img, text, (x,y), font, size, color, thickens
    #     cv2.putText(frame, str(round(cap.get(0)/1000, 2)) + 's',
    #                 (10, 15), font, 0.5, (255, 255, 255), 1)
    #     cv2.putText(frame, 'f.nr:' + str(cap.get(1)),
    #                 (100, 15), font, 0.5, (255, 255, 255), 1)
    #     cv2.imshow('frame', frame)

    cap.release()
    cv2.destroyAllWindows()
    return video_fragment


def read_image(path, name, ext, amount):
    """
    Function for reading images from folder. Name of images should be:
    name_index.extension so function can work automatic.
    Indexes should be in order! If they are not, function stops if image
    with next index is not found.
    Example: image_5.jpg -> read_image('path', 'image_', 'jpg', 50)
    :param path: string, path of images to read
    :param name: string, name of image without index
    :param ext: string, extension of image to read with ".", ex: '.jpg'
    :param amount: integer,
    :return: selected images as table if image exist or omits the image
    if it doesn't exist
    """
    images = []
    for i in range(amount):
        # try:
        print(path + '/' + name + str(i) + ext)
        img = cv2.imread(path + '/' + name + str(i) + ext, 1)
        # check if image was read
        try:
            if img.shape[0]:
                images.append(img)
        except AttributeError:
            pass
    return images


def blob_detect(img_with_blobs):
    params = cv2.SimpleBlobDetector_Params()
    # # Change thresholds
    # params.minThreshold = 10
    # params.maxThreshold = 200

    # Filter by Area.
    params.filterByArea = True
    params.minArea = 5

    # Filter by Circularity
    params.filterByCircularity = True
    params.minCircularity = 0.0

    # Filter by Convexity
    params.filterByConvexity = True
    params.minConvexity = 0.0

    # Filter by Inertia
    params.filterByInertia = True
    params.minInertiaRatio = 0.0

    # Create a detector with the parameters
    detector = cv2.SimpleBlobDetector_create(params)

    # get positions of blobs
    keypoints = detector.detect(img_with_blobs)

    # Draw detected blobs as red circles.
    # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the
    # circle corresponds to the size of blob
    im_with_keypoints = cv2.drawKeypoints(img_with_blobs, keypoints,
                                          np.array([]), (0, 0, 255),
                        cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    return im_with_keypoints


def get_log_kernel(siz, std):
    """
    LoG(x,y) =
(1/(pi*sigma^4)) * (1 - (x^2+y^2)/(sigma^2)) * (e ^ (- (x^2 + y^2) / 2sigma^2)

    :param siz:
    :param std:
    :return:
    """
    x = np.linspace(-siz, siz, 2*siz+1)
    y = np.linspace(-siz, siz, 2*siz+1)
    x, y = np.meshgrid(x, y)
    arg = -(x**2 + y**2) / (2*std**2)
    h = np.exp(arg)
    h[h < sys.float_info.epsilon * h.max()] = 0
    h = h/h.sum() if h.sum() != 0 else h
    h1 = h*(x**2 + y**2 - 2*std**2) / (std**4)
    return h1 - h1.mean()


def img_inv(img):
    """
    Return inversion of an image.
    :param img: Input image.
    :return: Inverted image.
    """
    return cv2.bitwise_not(img)


def local_maxima(gray_image):
    """
    Finds local maxima in grayscale image.
    source:
    https://dsp.stackexchange.com/questions/17932/finding-local-
    brightness-maximas-with-opencv
    :param gray_image: Input 2D image.
    :return: Coordinates of local maxima points.
    """

    square_diameter_log_3 = 3  # 27x27

    total = gray_image
    for axis in range(2):
        d = 1
        for k in range(square_diameter_log_3):
            total = np.maximum(total, np.roll(total, d, axis))
            total = np.maximum(total, np.roll(total, -d, axis))
            d *= 3
    # if total == gray_iamge, maxima = total
    maxima = total == gray_image
    h, w = gray_image.shape
    result = []
    for j in range(h):
        for k in range(w):
            # gray_image[j][k] has float values!
            if maxima[j][k] and gray_image[j][k]*255 > 1:
                # watch pixel coordinates output! (w, h)
                result.append((k, j))
    return result


def munkres(matrix):
    """
    Implementation of Hungarian algorithm for solving the Assignment Problem
    between measurements and estimates in multidimensional state observer.
    Example of usage:
        indexes = munkres(matrix)
    :param matrix: input matrix - should be a square matrix
    :return: index_list of tuples with assigned indexes
    """
    # cost matrix
    cost_matrix = []
    # create rows to write to
    for row in matrix:
        cost_row = []
    # write into rows
    for col in row:
        # cost_row += [sys.maxsize - col]
        cost_row += [col]
        cost_matrix += [cost_row]
    # print_matrix(cost_matrix, msg='Cost matrix:')
    m = Munkres()
    indexes = m.compute(cost_matrix)
    # print_matrix(matrix, msg='Highest profit through this matrix:')
    total = 0
    index_list = []
    for row, column in indexes:
        value = matrix[row][column]
        total += value
        index_list.append((row, column))
        # print('({}, {}) -> {}'.format(row, column, value))
    # print('total profit={}'.format(total))
    return index_list


# list of all VideoCapture methods and attributes
# [print(method) for method in dir(cap) if callable(getattr(cap, method))]

start_frame = 0
stop_frame = 500
font = cv2.FONT_HERSHEY_SIMPLEX
vid_fragment = select_frames('static/files/CIMG4027.MOV', start_frame,
                             stop_frame)

height = vid_fragment[0].shape[0]
width = vid_fragment[0].shape[1]
# kernel for morphological operations
# check cv2.getStructuringElement() doc for more info
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19))
erosion_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

i = 0
bin_frames = []
for frame in vid_fragment:
    if cv2.waitKey(15) & 0xFF == ord('q'):
        break
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    for m in range(height):  # height
        for n in range(width):  # width
            if n > 390 or m > 160:
                gray_frame[m][n] = 120

    # create a CLAHE object (Arguments are optional)
    # clahe = cv2.createCLAHE(clipLimit=8.0, tileGridSize=(8, 8))
    # cl1 = clahe.apply(gray_frame)
    ret, th1 = cv2.threshold(gray_frame, 60, 255, cv2.THRESH_BINARY)
    # frame_thresh1 = otsu_binary(cl1)
    bin_frames.append(th1)
    print(i)
    i += 1

i = 0
maxima_points = []
x = []
y = []
for frame in bin_frames:
    # prepare image - morphological operations
    erosion = cv2.erode(frame, erosion_kernel, iterations=1)
    opening = cv2.morphologyEx(erosion, cv2.MORPH_OPEN, kernel)
    dilate = cv2.dilate(opening, dilate_kernel, iterations=2)

    # create LoG kernel for finding local maximas
    log_kernel = get_log_kernel(30, 15)
    log_img = cv2.filter2D(dilate, cv2.CV_32F, log_kernel)

    # get local maximas of filtered image and append them to the maxima list
    maxima_points.append(local_maxima(log_img))

    frame_maximas = local_maxima(log_img)

    for point in frame_maximas:
        x.append(point[0])
        y.append(point[1])
        # print(point)
    print(i)
    i += 1

i = 0
for frame in vid_fragment:
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break
    # img, text, (x,y), font, size, color, thickens
    cv2.putText(frame, 'f.nr:' + str(start_frame + i + 1),
                (100, 15), font, 0.5, (254, 254, 254), 1)

    # mark local maximas for every frame
    for point in maxima_points[i]:
        cv2.circle(frame, point, 3, (0, 0, 255), 1)
    i += 1
    cv2.imshow('bin', frame)

# input('Press enter in the console to exit..')
cv2.destroyAllWindows()
# plot point by means of matplotlib (plt)
plt.plot(x, y, 'r.')
# # [xmin xmax ymin ymax]
plt.axis([0, width, height, 0])
plt.xlabel('width [px]')
plt.ylabel('height [px]')
plt.title('Objects past points (not trajectories)')
plt.grid()
plt.show()

# print(maxima_points)



