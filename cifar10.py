import argparse
import matplotlib
matplotlib.use("Agg")
import numpy as np
import os
from core.callbacks import TrainingMonitor
from core.nn import MiniGoogLeNet
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=70, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=5e-3, help="Initial learning rate")
    args = parser.parse_args()

    # Define learning rate scheduler function
    def poly_decay(epoch):
        power = 1.0
        lr = args.lr * (1 - (epoch / float(args.epochs))) ** power
        return lr

    (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    # Apply mean subtraction
    X_train = X_train.astype("float")
    X_test = X_test.astype("float")
    mean = np.mean(X_train, axis=0)
    X_train -= mean
    X_test -= mean

    # One hot encoding
    label_encoder = LabelBinarizer()
    y_train = label_encoder.fit_transform(y_train)
    y_test = label_encoder.transform(y_test)

    os.makedirs("output", exist_ok=True)

    # Training monitor
    fig_path = os.path.sep.join(["output", f"{os.getpid()}.png"])
    json_path = os.path.sep.join(["output", f"{os.getpid()}.json"])
    training_monitor = TrainingMonitor(fig_path, json_path=json_path)

    # Learning rate scheduler
    scheduler = LearningRateScheduler(poly_decay)

    # Image augmentation
    augmentation = ImageDataGenerator(
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        fill_mode="nearest"
    )

    optimizer = SGD(lr=args.lr, momentum=0.9)
    model = MiniGoogLeNet.build(32, 32, 3, 10)
    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])

    model.fit(
        augmentation.flow(X_train, y_train, batch_size=64),
        epochs=args.epochs,
        steps_per_epoch=len(X_train) // 64,
        validation_data=(X_test, y_test),
        callbacks=[training_monitor, scheduler],
        verbose=1
    )

    os.makedirs("models", exist_ok=True)
    model.save(os.path.sep.join(["models", f"{os.getpid()}.hdf5"]))


if __name__ == '__main__':
    main()
