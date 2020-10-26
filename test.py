import json
from config import config
from core.io import HDF5Reader
from core.metrics import accuracy
from core.preprocessing import MeanSubtraction
from core.preprocessing import Resize
from core.preprocessing import ToArray
from tensorflow.keras.models import load_model

def main():


    # Define preprocessors
    means = json.loads(open(config.MEAN_PATH).read())
    mean_subtraction = MeanSubtraction(means["R"], means["G"], means["B"])
    preprocessors = [Resize(64, 64), mean_subtraction, ToArray()]

    test_gen = HDF5Reader(
        config.TEST_PATH,
        batch_size=64,
        preprocessors=preprocessors,
        classes=config.NUM_CLASSES
    )

    model = load_model(config.MODEL_PATH)
    preds = model.predict(test_gen.generator(epochs=1), steps=test_gen.num_images // 64, max_queue_size=10)

    rank1, rank5 = accuracy(preds, test_gen.db["labels"])
    print(f"[INFO] Rank-1 accuracy: {rank1:.2f}")
    print(f"[INFO] Rank-5 accuracy: {rank5:.2f}")
    test_gen.close()

if __name__ == '__main__':
    main()
