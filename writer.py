import os 

import h5py as h5
from imutils import paths

from keras.preprocessing.image import load_img, img_to_array

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle


class FileAlreadyOpenError(RuntimeError):
    pass


class HDF5ImageWriter(object):
    def __init__(self, src, dims, X_key="images", y_key="labels", buffer_size=512):

        self.src: str = src
        self.dims = dims
        self.X_key: str = X_key
        self.y_key: str = y_key
        self.db = None
        self.images = None
        self.labels = None
        self.buffer_size = buffer_size
        self.buffer = {"tmp_images": [], "tmp_labels": []}
        self._index = 0

    def __enter__(self):
        if self.db is not None:
            raise FileAlreadyOpenError("The HDF5 file is already open!")

        self.db = h5.File(self.src, "w")
        self.images = self.db.create_dataset(self.X_key, self.dims, dtype="float32")
        self.labels = self.db.create_dataset(self.y_key, (self.dims[0],), dtype="uint8")

        return self

    def __exit__(self, type_, value, traceback):
        self.__close()

    def add(self, images, labels):
        self.buffer["tmp_images"].extend(images)
        self.buffer["tmp_labels"].extend(labels)

        if len(self.buffer["tmp_images"]) >= self.buffer_size:
            self.__flush()
            
    def add_classes(self, classes):
        datatype = h5.string_dtype(encoding="utf-8")
        classes_set = self.db.create_dataset("classes", (len(classes),), dtype=datatype)
        classes_set[:] = classes
        
        print('[Classes] Added', (len(classes)))

    def __flush(self):
        index = self._index + len(self.buffer["tmp_images"])
        self.images[self._index : index] = self.buffer["tmp_images"]
        self.labels[self._index : index] = self.buffer["tmp_labels"]
        self._index = index

        self.buffer = {"tmp_images": [], "tmp_labels": []}

    def __close(self):
        if len(self.buffer["tmp_images"]) > 0:
            self.__flush()

        self.db.close()
        
# dataset source
directory = './dataset/Flickr/'

# get image paths
X_paths = shuffle(list(paths.list_images(directory)))

# get image classes
classes = [path.split(os.path.sep)[-1].split('_')[0] for path in X_paths]

# encode classes  str => int
enc = LabelEncoder()
y = enc.fit_transform(classes)

# randomized split
X_train, X_test, y_train, y_test = train_test_split(X_paths, y, test_size=0.2, random_state=42)

# set writer
h5_writer = HDF5ImageWriter(
    src="train.h5", dims=(len(X_train), 150, 150, 3)
)

import efficientnet.keras as efn

# build hdf5
with h5_writer as writer:
    for index, (path, label) in enumerate(zip(X_train, y_train)):
        raw_image = load_img(path)
        image = img_to_array(raw_image)
        # efficient net resizing b0: 150, b1: 220, b2: 240, b3: 300
        image = efn.center_crop_and_resize(image, 150)
        # mean scaling (disable -1/1)
        image = efn.preprocess_input(image)
        writer.add([image], [label])
        print(index, 'over', len(X_train), 'Added', path, label)
    print(enc.classes_)
    writer.add_classes(enc.classes_)