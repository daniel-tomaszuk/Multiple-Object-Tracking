from django.shortcuts import render
import numpy as np
from matplotlib import pyplot as plt
import cv2
import sys
from scipy import ndimage


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



# list of all VideoCapture methods and attributes
# [print(method) for method in dir(cap) if callable(getattr(cap, method))]

start_frame = 1000
stop_frame = 1500
font = cv2.FONT_HERSHEY_SIMPLEX
vid_fragment = select_frames('static/files/CIMG4027.MOV', start_frame,
                             stop_frame)
# kernel for morphological operations

# el = ndimage.generate_binary_structure(2, 1)
# kernel = np.ones((5, 5), np.uint8)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19))
erosion_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
i = 0
bin_frames = []
for frame in vid_fragment:
    if cv2.waitKey(15) & 0xFF == ord('q'):
        break
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    for m in range(gray_frame.shape[0]):  # height
        for n in range(gray_frame.shape[1]):  # width
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
for frame in bin_frames:
    # img, text, (x,y), font, size, color, thickens

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break
    # cv2.imshow('frame', frame_thresh1)
    erosion = cv2.erode(frame, erosion_kernel, iterations=1)
    opening = cv2.morphologyEx(erosion, cv2.MORPH_OPEN, kernel)
    dilate = cv2.dilate(opening, dilate_kernel, iterations=2)

    cv2.putText(dilate, 'f.nr:' + str(start_frame + i + 1),
                (100, 15), font, 0.5, (0, 0, 0), 1)
    cv2.imshow('bin', dilate)
    i += 1

# input('Press enter in the console to exit..')
cv2.destroyAllWindows()




