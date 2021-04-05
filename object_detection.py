# img_mat: image matrix

import os
from pathlib import Path
import pickle
import sys

import cv2 as cv
import face_recognition as fr
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


def create_face_encoding():
    """
    - Load images from a directory with training images.
    - Generate encoding for each face.
    - Persist the encoding and the name of the person as a pickle file.
    - Each training image has a single human face.
    - Name of the file is the name of the person in the image.
    """
    img_file_extns = ('.jpeg', '.jpg', '.png', '.gif', '.tiff')
    encodings_dir_path = Path('./data/train_faces/face_encodings')
    encodings_dir_path.mkdir(exist_ok=True)
    # TODO: test that training directory exists and is not empty
    with os.scandir("./data/train_faces") as dir:
        for entry in dir:
            if entry.is_file() and entry.name.lower().endswith(img_file_extns):
                face_name = os.path.splitext(entry.name)[0]
                img_mat = fr.load_image_file(entry.path)
                face_encoding = fr.face_encodings(img_mat)[0]
                out_filepath = encodings_dir_path / f'{face_name}.pkl'
                with open(out_filepath, 'wb') as f:
                    pickle.dump(face_encoding, f)


def load_face_encodings():
    name_encoding_map = dict()
    encodings_dir_path = Path('./data/train_faces/face_encodings')
    if encodings_dir_path.is_dir():
        with os.scandir(encodings_dir_path) as dir:
            for entry in dir:
                if entry.is_file() and entry.name.lower().endswith('.pkl'):
                    face_name = os.path.splitext(entry.name)[0]
                    with open(entry.path, 'rb') as f:
                        name_encoding_map[face_name] = pickle.load(f)
    return name_encoding_map


def main():
    # img_mat = read_img_from_file(file_name='./data/puppy.jpg')
    # display_img(img_mat, delay_in_millsec=4000)
    # display_video_from_cam()
    # create_face_encoding()
    name_encoding_map = load_face_encodings()
    # test_data_path = Path('./data/test_imgs')
    # test_img = test_data_path / 'u11.jpg'


if __name__ == '__main__':
    main()
