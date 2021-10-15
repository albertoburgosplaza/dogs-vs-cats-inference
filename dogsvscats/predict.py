from typing import Union
import requests
from PIL import Image
import numpy as np
import onnxruntime

CLASSES = ["cat", "dog"]


def predict(image_path: str):
    image = Image.open(image_path)
    image = image.resize((224, 224), resample=Image.BILINEAR)  # W, H
    image = np.array(image, dtype=np.float32)
    image /= 255
    image = (image - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    image = image.astype(np.float32)
    image = np.transpose(image, (2, 0, 1))
    image = np.expand_dims(image, axis=0)  # Needs 1 x 3 x H x W

    ort_session = onnxruntime.InferenceSession(
        "/home/alberto/workspace/dogs-vs-cats-inference/data/mobilenet_v3_small.onnx"
    )
    ort_inputs = {ort_session.get_inputs()[0].name: image}
    ort_outs = ort_session.run(None, ort_inputs)

    return CLASSES[np.argmax(ort_outs[0][0])]
