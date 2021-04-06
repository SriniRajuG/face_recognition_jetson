# img_mat: image matrix

import os
from pathlib import Path
import pickle
import sys
from typing import List, Tuple

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


def load_face_encodings() -> Tuple[List[np.ndarray], List[str]]:
    """
    Iterate over and load .pkl files in a specific directory
    Each .pkl file has face encoding of a person.
    Name of the .pkl file is the name of the person.
    """
    known_encodings = list()
    known_names = list()
    encodings_dir_path = Path('./data/train_faces/face_encodings')
    if encodings_dir_path.is_dir():
        with os.scandir(encodings_dir_path) as dir:
            for entry in dir:
                if entry.is_file() and entry.name.lower().endswith('.pkl'):
                    face_name = os.path.splitext(entry.name)[0]
                    with open(entry.path, 'rb') as f:
                        known_encodings.append(pickle.load(f))
                        known_names.append(face_name)
    return (known_encodings, known_names)


def get_face_match_names(
        test_img_face_locations: List[Tuple[int]],
        test_img_face_encodings: List[np.ndarray],
        known_encodings: List[np.ndarray],
        known_names: List[str],
        ):
    test_img_face_names = list()
    for _, face_encoding in zip(
            test_img_face_locations,
            test_img_face_encodings,
            ):
        face_name = 'Unknown'
        matches = fr.compare_faces(known_encodings, face_encoding)
        if True in matches:
            first_match_index = matches.index(True)
            face_name = known_names[first_match_index]
        test_img_face_names.append(face_name)
    return test_img_face_names


def draw_rectangles_on_faces(img_mat, face_locations, face_names):
    font = cv.FONT_HERSHEY_SIMPLEX
    for (top, right, bottom, left), face_name in zip(
            face_locations,
            face_names,
            ):
        blue = (255, 0, 0)
        yellow = (0, 255, 255)
        cv.rectangle(img_mat, pt1=(left, top), pt2=(right, bottom), 
            color=blue, thickness=2)
        cv.rectangle(img_mat, pt1=(left, top), pt2=(left + 200, top + 30), 
            color=yellow, thickness=-1)
        cv.putText(img_mat, text=face_name, org=(left, top + 20), 
            fontScale=0.75, fontFace=font, color=blue, thickness=2)
    display_img(img_mat=img_mat, delay_in_millsec=1000)


def main():
    # img_mat = read_img_from_file(file_name='./data/puppy.jpg')
    # display_img(img_mat, delay_in_millsec=4000)
    # display_video_from_cam()
    # create_face_encoding()
    known_encodings, known_names = load_face_encodings()
    test_data_dir_path = Path('./data/test_imgs')
    test_img_path = test_data_dir_path / 'u11.jpg'
    test_img_mat = fr.load_image_file(test_img_path)
    test_img_mat = cv.cvtColor(test_img_mat, cv.COLOR_BGR2RGB)
    test_img_face_locations = fr.face_locations(test_img_mat)
    test_img_face_encodings = fr.face_encodings(
        test_img_mat,
        test_img_face_locations,
    )
    test_img_face_names = get_face_match_names(
        test_img_face_locations,
        test_img_face_encodings,
        known_encodings,
        known_names,
    )
    draw_rectangles_on_faces(
        test_img_mat,
        test_img_face_locations,
        test_img_face_names,
    )


if __name__ == '__main__':
    main()
