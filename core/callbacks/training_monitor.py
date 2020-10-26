import json
import os
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.callbacks import BaseLogger

class TrainingMonitor(BaseLogger):

    def __init__(self, fig_path, json_path=None, start=0):
        super(TrainingMonitor, self).__init__()
        self.fig_path = fig_path
        self.json_path = json_path
        self.start = start

    def on_train_begin(self, logs={}):
        self.H = {}
        if self.json_path:
            if os.path.exists(self.json_path):
                self.H = json.loads(open(self.json_path).read())
                if self.start > 0:
                    for key in self.H.keys():
                        self.H[key] = self.H[key][:self.start]

    def on_epoch_end(self, epoch, logs={}):
        for key, value in logs.items():
            data = self.H.get(key, [])
            data.append(float(value))
            self.H[key] = data

        if self.json_path:
            file = open(self.json_path, "w")
            file.write(json.dumps(self.H))
            file.close()

        N = np.arange(0, len(self.H["loss"]))
        plt.style.use("ggplot")
        plt.figure()
        plt.plot(N, self.H["loss"], label="Training loss")
        plt.plot(N, self.H["val_loss"], label="Validation loss")
        plt.plot(N, self.H["accuracy"], label="Training accuracy")
        plt.plot(N, self.H["val_accuracy"], label="Validation accuracy")
        plt.title(f"Training loss [Epoch {len(self.H['loss'])}]")
        plt.xlabel("Epochs")
        plt.ylabel("Loss/Accuracy")
        plt.legend()
        plt.savefig(self.fig_path)
        plt.close()
