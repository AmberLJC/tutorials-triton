


Convert:
```
docker run -it --gpus all -v ${PWD}:/workspace nvcr.io/nvidia/pytorch:23.05-py3
pip install transformers
python convert.py
```

Build mode directory:
```commandline
mkdir model_repository
cd model_repository
mkdir bert
cd bert
mkdir 1
cd ../../
mv bert.onnx model_repository/bert/1/model.onnx
mv config.pbtxt model_repository/bert/config.pbtxt
```
> exit docker

Start Triton:
```
docker run --gpus all --rm -p 8000:8000 -p 8001:8001 -p 8002:8002 -v ${PWD}/model_repository:/models nvcr.io/nvidia/tritonserver:23.05-py3 tritonserver --model-repository=/models
``

Client request
```

docker run -it --net=host -v ${PWD}:/workspace/ nvcr.io/nvidia/tritonserver:23.05-py3-sdk bash
pip install torch
python client.py

``