# Posenet TfJs2TFlite

This repository provides ready to use script to convert Posenet Tensorflow JS models to Tensorflow Lite model.

## How to use ?

1) Modify config.yml

```
Currently script can build 3 different achitectures of MobileNetv1 models.
Modify 'checkpoints_index' parameter in config.yml to change the architecture.
Similarly modify 'outputStride' and 'image_size' parameters as required.
```

> Each parameter effects the accuracy and speed of detection. 
To know more visit [Posenet-TfJs](https://github.com/tensorflow/tfjs-models/tree/master/posenet#config-params-in-posenetload)

2) Run

```
$ python tfjs2tflite.py
```

3) Output
```
find the model in _models/ folder
```
