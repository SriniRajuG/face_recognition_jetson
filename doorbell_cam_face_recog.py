# gst: gstreamer
# img: image

import datetime as dt
import os
from pathlib import Path
import pickle
import sys
from typing import List, Tuple, Dict

import cv2 as cv
import face_recognition as fr
import numpy as np


def create_face_encoding():
    """
    - Load images from a directory with training images.
    - Generate encoding for each face.
    - Persist the encodings and metadatas as a pickle file.
    """
    img_file_extns = ('.jpeg', '.jpg', '.png', '.gif', '.tiff')
    out_filepath = Path('./data/pi_cam/known_faces.pkl')
    # TODO: check that training directory exists and is not empty
    known_encodings = list()
    known_metadatas = list()
    with os.scandir("./data/pi_cam/train_faces") as dir:
        for entry in dir:
            if entry.is_file() and entry.name.lower().endswith(img_file_extns):
                face_label = os.path.splitext(entry.name)[0]
                img = fr.load_image_file(entry.path)
                face_encoding = fr.face_encodings(img)[0]
                small_img = cv.resize(img, (150, 150))
                known_encodings.append(face_encoding)
                known_metadatas.append({
                    'face_img': small_img,
                    'face_label': face_label,
                })
    with open(out_filepath, 'wb') as f:
        pickle.dump((known_encodings, known_metadatas), f)


def load_face_encodings() -> Tuple[List[np.ndarray], List[Dict]]:
    encodings_file_path = Path('./data/pi_cam/known_faces.pkl')
    try:
        with open(encodings_file_path, 'rb') as f:
            known_face_encodings, known_face_metadata = pickle.load(f)
        print('Known faces info. loaded from database')
    except FileNotFoundError as e:
        print(e)
        print('----')
        print('No data related to known faces is found.')
        known_face_encodings, known_face_metadata = list(), list()
    finally:
        return (known_face_encodings, known_face_metadata)


def lookup_known_faces(face_encoding, known_encodings, known_metadata):
    metadata = None
    if len(known_encodings) == 0:
        return metadata
    face_distances = fr.face_distance(known_encodings, face_encoding)
    best_match_idx = np.argmin(face_distances)
    if face_distances[best_match_idx] < 0.65:
        metadata = known_metadata[best_match_idx]
        if 'first_seen_time' not in metadata:
            metadata['first_seen_time'] = dt.datetime.now()
            metadata['new_interaction_start_time'] = dt.datetime.now()
        metadata['last_seen_time'] = dt.datetime.now()
        # is_new_interaction = (
        #     (dt.datetime.now() - metadata['new_interaction_start_time']) >
        #     dt.timedelta(seconds=3)
        # )
        # if is_new_interaction:
        #     metadata['new_interaction_start_time'] = dt.datetime.now()
        #     metadata['n_interactions'] += 1
    return metadata


def register_new_face(face_encoding, face_img, known_encodings, known_metadatas):
    known_encodings.append(face_encoding)
    known_metadatas.append({
        'first_seen_time': dt.datetime.now(),
        'new_interaction_start_time': dt.datetime.now(),
        'last_seen_time': dt.datetime.now(),
        # 'n_interactions': 1,
        'face_img': face_img,
        'face_label': None,
    })


def save_known_faces(known_encodings, known_metadatas):
    encodings_file_path = './data/known_faces.dat'
    with open(encodings_file_path, 'wb') as f:
        face_data = [known_encodings, known_metadatas]
        pickle.dump(face_data, f)
        print('Known faces backed up to disk.')


def get_gst_pipe():
    width = 800
    height = 600
    flip = 2
    gstreamer_pipe = f" \
        nvarguscamerasrc sensor-id=0 ! \
        video/x-raw(memory:NVMM), width=3264, height=2464, framerate=21/1, \
            format=NV12 ! \
        nvvidconv flip-method={flip} ! \
        video/x-raw, width={width}, height={height}, format=BGRx ! \
        videoconvert ! \
        video/x-raw, format=BGR ! \
        appsink \
        "
    return gstreamer_pipe


def main():
    n_faces_since_save = 0
    img_scale_frac = 0.25
    red = (0, 0, 255)
    white = (255, 255, 255)
    label_font = cv.FONT_HERSHEY_DUPLEX
    known_encodings, known_metadatas = load_face_encodings()
    gst_pipe = get_gst_pipe()
    # vid_cap = cv.VideoCapture('./data/door.mp4')
    vid_cap = cv.VideoCapture(gst_pipe)
    while True:
        ret_val, img = vid_cap.read()
        if not ret_val:
            break
        small_img = cv.resize(src=img, dsize=(0, 0), fx=img_scale_frac, fy=img_scale_frac)
        small_img = cv.cvtColor(small_img, cv.COLOR_BGR2RGB)
        face_locations = fr.face_locations(small_img)
        face_encodings = fr.face_encodings(small_img, face_locations)
        face_labels = list()
        for face_location, face_encoding in zip(face_locations, face_encodings):
            metadata = lookup_known_faces(
                face_encoding,
                known_encodings,
                known_metadatas,
                )
            if metadata is not None:  # Found a match
                # interaction_duration = dt.datetime.now() - metadata['new_interaction_start_time']
                # face_label = f"At door {int(interaction_duration.total_seconds())}s"
                face_label = f"{metadata['face_label']}"
            else:
                face_label = 'New visitor'
                top, right, bottom, left = face_location
                face_img = img[top:bottom, left:right]
                face_img = cv.resize(face_img, (150, 150))
                register_new_face(face_encoding, face_img, known_encodings, known_metadatas)
            face_labels.append(face_label)
        for face_location, face_labels in zip(face_locations, face_labels):
            inv_img_scale_frac = int(1 / img_scale_frac)
            top, right, bottom, left = [loc * inv_img_scale_frac for loc in face_location]
            cv.rectangle(img, pt1=(left, top), pt2=(right, bottom), color=red, thickness=2)
            cv.rectangle(img, (left, bottom - 35), (right, bottom), red, cv.FILLED)
            cv.putText(img, face_label, (left + 6, bottom - 6), label_font, fontScale=0.8, color=white, thickness=1)
        cv.imshow('Video', img)
        retkey_ordval = cv.waitKey(delay=1)
        if retkey_ordval == ord('q'):
            save_known_faces(known_encodings, known_metadatas)
            break
    vid_cap.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    # create_face_encoding()
    main()
