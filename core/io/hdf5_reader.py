import h5py
import numpy as np
from tensorflow.keras.utils import to_categorical

class HDF5Reader:

    def __init__(self, filename, batch_size, preprocessors=None, augmentation=None, binarize=True, classes=2):
        self.batch_size = batch_size
        self.preprocessors = preprocessors
        self.augmentation = augmentation
        self.binarize = binarize
        self.classes = classes

        self.db = h5py.File(filename, "r")
        self.num_images = self.db["labels"].shape[0]

    def generator(self, epochs=np.inf):
        epoch = 0
        while epoch < epochs:

            for i in np.arange(0, self.num_images, self.batch_size):

                batch_images = self.db["data"][i:i + self.batch_size]
                batch_labels = self.db["labels"][i:i + self.batch_size]

                # Apply one hot encoding
                if self.binarize:
                    batch_labels = to_categorical(batch_labels, self.classes)

                # Preprocess images
                if self.preprocessors:
                    preprocessed = []
                    for image in batch_images:
                        for preprocessor in self.preprocessors:
                            image = preprocessor.preprocess(image)
                        preprocessed.append(image)
                    batch_images = np.array(preprocessed)

                # Apply data augmentation
                if self.augmentation:
                    batch_images, batch_labels = next(self.augmentation.flow(
                        batch_images, batch_labels, batch_size=self.batch_size
                    ))

                yield (batch_images, batch_labels)
            epoch += 1

    def close(self):
        self.db.close()
