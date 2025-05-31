# FSRE-HDN
This repo contains the code and data for **HyperNetwork-based Decoupling to Improve Model Generalization for Few-Shot Relation Extraction**
(EMNLP-2023).


## Requirements
* Python 3.7.4
* PyTorch 1.7.0
* CUDA 10.2


## Dataset
The expected structure of files is:
```

 |-- checkpoint
 |-- data
 |-- fewshot_re_kit
 |-- logs
 |-- models
 |-- train_demo.py
 |-- run_bert.sh
```

## Code
Our model **HND** is placed in the ``HND.py`` file.

Our **two-stage training strategy** can be found on lines 220 to 320 of the ``framework.py`` file in the ``fewshot_re_kit`` folder.



## Training
You can train a N-way-K-shot model by:
```
sh run_bert.sh
```
In the file ``run_bert.sh``, we can modify the settings and hyper-parameters of the model.
