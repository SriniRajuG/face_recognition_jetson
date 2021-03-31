# img_mat: image matrix

import sys

import cv2 as cv
import numpy as np


def read_img_from_file(file_name: str) -> np.ndarray:
    img_mat = cv.imread(file_name)
    if img_mat is None:
        sys.exit('Could not read image')
    return img_mat


def display_img(
        img_mat,
        window_name: str = '',
        delay_in_millsec: int = 0,
        ) -> int:
    cv.imshow(winname=window_name, mat=img_mat)
    retkey_ordval = cv.waitKey(delay=delay_in_millsec)
    return retkey_ordval


def display_video_from_cam() -> None:
    videocap = cv.VideoCapture(index=0)
    if not videocap.isOpened():
        sys.exit('Video capture is not sucessful.')
    while True:
        _, frame = videocap.read()
        retkey_ordval = display_img(img_mat=frame, delay_in_millsec=1)
        if retkey_ordval == ord('q'):
            break
    videocap.release()
    cv.destroyAllWindows()


def main():
    # img_mat = read_img_from_file(file_name='./data/puppy.jpg')
    # display_img(img_mat, delay_in_millsec=4000)
    display_video_from_cam()


if __name__ == '__main__':
    main()
