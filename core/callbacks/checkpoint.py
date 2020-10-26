import os
from tensorflow.keras.callbacks import Callback

class Checkpoint(Callback):

    def __init__(self, output_dir, interval=5, start=0):
        super(Callback, self).__init__()
        self.output_dir = output_dir
        self.interval = interval
        self.start = start

    def on_epoch_end(self, epoch, logs={}):
        if (self.start + 1) % self.interval == 0:
            filename = os.path.sep.join([self.output_dir, f"epoch_{self.start + 1}.hdf5"])
            self.model.save(filename, overwrite=True)
        self.start += 1
