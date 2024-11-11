# DEVNET Project 2: Health Detection via Image Classification

## Installation

Installation is very simple, just install the needed dependencies.

```shell
python3 -m pip install -r requirements.txt
```

## Training

While the included `model.pth` weights file should be enough for most,
you can train the model from scratch or further with the `train.py` and `pretrain.py` file.
The script will download the kagglehub dataset and train based on those images.

To start, we need to preprocess our images:

```shell
python3 pretrain.py
```

Now we can train the model:

```shell
python3 train.py
```

### Training Options:

The training script will automatically use the best device available,
and you can tweak the epoch and batch size.

```shell
$ python train.py -h
usage: Trainer [-h] [-e EPOCHS] [-b BATCH] [-s]

Trains the model based on preprocessed images

options:
  -h, --help            show this help message and exit
  -e EPOCHS, --epochs EPOCHS
                        The number of epochs to train on
  -b BATCH, --batch BATCH
                        The size of the batches
  -s, --silent          Disables all print statements
```

After training there will be a new or modified file `model.pth` which contains the weights for the model.

## Running The Model

### Via Premade File

The script `test.py` is made to easily test the model on supplied images:

```shell
$ python test.py -h
usage: test.py [-h] [-p PATH] [-w WEIGHTS] [--debug]

options:
  -h, --help            show this help message and exit
  -p PATH, --path PATH  The path to the image or image directory
  -w WEIGHTS, --weights WEIGHTS
                        The path to the weights of the model
  --debug               Whether to enable very verbose and helpful debugging
```

### Via API

There is also an api in the `HealthModel.py` file which allow you to
access an interposer and evaluate batches or urls to images.

```python
import cv2
from HealthModel import HealthModel

model = HealthModel("model.pth")

image = cv2.imread("PATH TO IMAGE FILE")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
pred = model.predict(image, multi_leaf=..., _debug=...)

print(f"The image is {pred[0].upper()} ({pred[1] * 100:.2f}%)")
```

The `.predict()` method will return the health of the plant as a string (`"healthy"` or `"unhealthy"`)
and a confidence value (percent from `0.0` to `1.0`).

## Web UI

...