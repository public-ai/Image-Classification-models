import pandas as pd
import os
import numpy as np
import cv2
from urllib.request import urlretrieve
import zipfile
import time
import sys

dirnames = []
for dirname in os.getcwd().split('/'):
    dirnames.append(dirname)
    if dirname == 'tinyimagenet':
        break
DATA_DIR = "/".join(dirnames+['data'])
DOWNLOAD_URL = 'https://s3.ap-northeast-2.amazonaws.com/pai-datasets/alai-deeplearning/tinyimagenet.zip'


class Dataset:
    def __init__(self, data_type='train'):
        np.random.seed(1)
        self._df = load_dataset(data_type)
        self._label_map = load_label_map()

        self._image_dir = os.path.join(DATA_DIR,
                                       "{}/images/".format(data_type))
        self.num_data = len(self._df)
        self.num_classes = len(self._df.label.unique())
        self.shuffle()

        self._counter = 0

    def __len__(self):
        return self.num_data

    def __getitem__(self, index):
        if isinstance(index, int):
            batch = self._df.iloc[index]
            image = self._get_image(batch.filename)
            return image, batch.label
        else:
            batch = self._df.iloc[index]
            images = np.zeros((len(batch),64,64,3),dtype=np.uint8)
            for idx, (_, row) in enumerate(batch.iterrows()):
                images[idx] = self._get_image(row.filename)
            labels = batch.label.values
            return images, labels

    def _get_image(self, filename):
        image_path = os.path.join(self._image_dir, filename)
        image = cv2.imread(image_path)
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def label2name(self, label):
        if isinstance(label, int):
            return self._label_map[label]
        else:
            return pd.Series(label).map(self._label_map).values

    def next_batch(self, batch_size):
        if self._counter + batch_size >= self.num_data:
            self.shuffle()
            self._counter = 0
        self._counter += batch_size
        return self[self._counter-batch_size:self._counter]

    def shuffle(self):
        self._df = self._df.sample(frac=1).reset_index(drop=True)


def load_dataset(data_type):
    if data_type != 'train' and data_type !='validation':
        return ValueError("data type은 train / validation 중 하나입니다.")
    data_path = os.path.join(DATA_DIR,
                             '{}/data.csv'.format(data_type))
    if not os.path.exists(data_path):
        download_dataset()
    df = pd.read_csv(data_path)
    return df


def load_label_map():
    data_path = os.path.join(DATA_DIR,'code_map.csv')
    df = pd.read_csv(data_path,index_col=0)
    return df.name.to_dict()


def download_dataset():
    path = 'dataset.zip'
    urlretrieve(DOWNLOAD_URL, path, reporthook)
    print("Start to extract zip file...")
    with zipfile.ZipFile(path,'r') as f:
        f.extractall(DATA_DIR)
    print("end")


def reporthook(count, block_size, total_size):
    global start_time
    if count == 0:
        start_time = time.time()
        return
    duration = time.time() - start_time
    progress_size = int(count * block_size)
    speed = int(progress_size / (1024 * duration))
    percent = min(int(count * block_size * 100 / total_size),100)
    sys.stdout.write("\r...%d%%, %d MB, %d KB/s, %d seconds passed" %
                    (percent, progress_size / (1024 * 1024), speed, duration))
    sys.stdout.flush()
