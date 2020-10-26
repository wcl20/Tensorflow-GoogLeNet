import h5py
import os

class HDF5Writer:

    def __init__(self, filename, dims, buffer_size=1000):

        if os.path.exists(filename):
            raise ValueError(f"File {filename} already exist. Please remove file before continuing.")

        self.db = h5py.File(filename, "w")
        # Create two datasets: (1) Image (2) Labels
        self.data = self.db.create_dataset("data", dims, dtype="float")
        self.labels = self.db.create_dataset("labels", (dims[0], ), dtype="int")
        # Size of memory buffer
        self.buffer_size = buffer_size
        self.buffer = { "data": [], "labels": [] }
        self.idx = 0

    def add(self, data, labels):
        self.buffer["data"].extend(data)
        self.buffer["labels"].extend(labels)
        # Flush buffer to HDF5 file
        if len(self.buffer["data"]) >= self.buffer_size:
            self.flush()

    def flush(self):
        # Number of data in buffer
        n = len(self.buffer["data"])
        # Write to file
        self.data[self.idx:self.idx + n] = self.buffer["data"]
        self.labels[self.idx:self.idx + n] = self.buffer["labels"]
        # Update pointer
        self.idx = self.idx + n
        # Clear buffer
        self.buffer = { "data": [], "labels": [] }

    def store_class_names(self, class_names):
        dtype = h5py.special_dtype(vlen=str)
        self.class_names = self.db.create_dataset("class_names", (len(class_names), ), dtype=dtype)
        self.class_names[:] = class_names

    def close(self):
        if len(self.buffer["data"]) > 0:
            self.flush()
        self.db.close()
