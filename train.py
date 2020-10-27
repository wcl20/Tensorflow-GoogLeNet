import argparse
import json
import matplotlib
matplotlib.use("Agg")
import os
from config import config
from core.callbacks import TrainingMonitor
from core.callbacks import Checkpoint
from core.io import HDF5Reader
from core.nn import DeepGoogLeNet
from core.preprocessing import MeanSubtraction
from core.preprocessing import Resize
from core.preprocessing import ToArray
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None, help="Path to specific model")
    parser.add_argument("--start", type=int, default=0, help="Epoch to restart training")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs to train")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    args = parser.parse_args()

    # Define preprocessors
    means = json.loads(open(config.MEAN_PATH).read())
    mean_subtraction = MeanSubtraction(means["R"], means["G"], means["B"])
    preprocessors = [Resize(64, 64), mean_subtraction, ToArray()]

    # Define image augmentation
    augmentation = ImageDataGenerator(
        rotation_range=18,
        zoom_range=0.15,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        horizontal_flip=True,
        fill_mode="nearest"
    )

    # Data generator from HDF5 files
    train_gen = HDF5Reader(config.TRAIN_PATH,
        batch_size=64,
        preprocessors=preprocessors,
        augmentation=augmentation,
        classes=config.NUM_CLASSES
    )

    valid_gen = HDF5Reader(config.VALID_PATH,
        batch_size=64,
        preprocessors=preprocessors,
        classes=config.NUM_CLASSES
    )

    # Build/Load model
    if args.model:
        print(f"[INFO] Loading model from {args.model} ...")
        model = load_model(args.model)

        # Update learning rate
        print(f"[INFO] Old learning rate: {K.get_value(model.optimizer.lr)}")
        K.set_value(model.optimizer.lr, args.lr)
        print(f"[INFO] New learning rate: {K.get_value(model.optimizer.lr)}")

    else:
        print("[INFO] Compiling model ...")
        optimizer = Adam(lr=args.lr)
        model = DeepGoogLeNet.build(64, 64, 3, classes=config.NUM_CLASSES, reg=0.0002)
        model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])

    # Training monitor
    training_monitor = TrainingMonitor(config.FIG_PATH, json_path=config.JSON_PATH, start=args.start)
    # Check point
    os.makedirs(f"{config.OUTPUT_PATH}/checkpoints", exist_ok=True)
    checkpoint = Checkpoint(f"{config.OUTPUT_PATH}/checkpoints", start=args.start)
    callbacks = [training_monitor, checkpoint]

    model.fit(
        train_gen.generator(),
        epochs=args.epochs,
        steps_per_epoch=train_gen.num_images // 64,
        validation_data=valid_gen.generator(),
        validation_steps=valid_gen.num_images // 64,
        max_queue_size=10,
        callbacks=callbacks,
        verbose=1
    )

    train_gen.close()
    valid_gen.close()


if __name__ == '__main__':
    main()
