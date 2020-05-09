import scipy
import numpy as np
import scipy.signal
from rotate_img import rotate_image


# Align image so lines are as horizontal as possible
def align_image(img):
    img = 255 - img

    rotations = range(-20, 20, 2)
    highest_peak = -1
    best_rotation_data = -1
    best_rotation_img = -1
    best_angle = -1
    for angle in rotations:
        rotated_img = rotate_image(img, angle)
        img_row_sum = np.sum(rotated_img, axis=1)
        peak = max(img_row_sum)
        if peak > highest_peak:
            highest_peak = peak
            best_rotation_data = img_row_sum
            best_rotation_img = rotated_img
            best_angle = angle

    best_rotation_img = 255 - best_rotation_img
    return best_rotation_img, best_rotation_data, best_angle


def get_lines(img):
    img_row_sum = np.sum(255 - img, axis=1)
    total = img_row_sum.sum()

    peaks, _ = scipy.signal.find_peaks(img_row_sum,
                                       # height=40000,
                                       distance=50,
                                       prominence=max(total / 3000, 18000)
                                       )

    return peaks
