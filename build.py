import cv2
import os
import json
import numpy as np
import tqdm
from config import config
from core.io import HDF5Writer
from imutils import paths
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def main():

    # Load training images
    img_paths = list(paths.list_images(config.TRAIN_IMAGES_PATH))
    label_encoder = LabelEncoder()
    labels = [img_path.split(os.path.sep)[-3] for img_path in img_paths]
    labels = label_encoder.fit_transform(labels)
    train_paths, test_paths, y_train, y_test = train_test_split(img_paths, labels, test_size=config.NUM_TEST_IMAGES, stratify=labels, random_state=42)

    # Load validation images
    mappings = open(config.VALID_MAPPING_PATH).read().strip().split("\n")
    mappings = [row.split("\t")[:2] for row in mappings]
    valid_paths = [os.path.sep.join([config.VALID_IMAGES_PATH, mapping[0]]) for mapping in mappings]
    y_valid = label_encoder.transform([mapping[1] for mapping in mappings])

    datasets = [
        ("train", train_paths, y_train, config.TRAIN_PATH),
        ("valid", valid_paths, y_valid, config.VALID_PATH),
        ("test", test_paths, y_test, config.TEST_PATH)
    ]

    R, G, B = [], [], []
    for type, img_paths, labels, output_path in datasets:
        print(f"[INFO] Building {output_path} ...")
        print(f"[INFO] Number of images: {len(img_paths)}. Labels: {labels.shape}")

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        dataset = HDF5Writer(output_path, (len(img_paths), 64, 64, 3))

        for img_path, label in tqdm.tqdm(zip(img_paths, labels)):
            image = cv2.imread(img_path)
            if type == "train":
                b, g, r, = cv2.mean(image)[:3]
                B.append(b)
                G.append(g)
                R.append(r)
            dataset.add([image], [label])
        dataset.close()

    os.makedirs(config.OUTPUT_PATH, exist_ok=True)
    file = open(config.MEAN_PATH, "w")
    file.write(json.dumps({ "R": np.mean(R), "G": np.mean(G), "B": np.mean(B) }))
    file.close()

if __name__ == '__main__':
    main()
