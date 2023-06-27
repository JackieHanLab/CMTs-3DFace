## _Privacy-Preserving Age Estimation from Human 3D Facial Data with Coordinate-Wise Monotonic Transformations_ by Xinyu Bruce Yang, Jing-Dong Jackie Han _et al_.

![Face Transformer](/Face Transformer.png)

# Facial point cloud-based age estimation
Use scripts in `FPCT-age`. Put your data split record (simply in txt format) in `DataSplit`. Modify `config.yaml` in `config` for your training specifications. Run `script_main.py` with `hydra`-style parameter passing (https://hydra.cc/docs/intro/) to train. Modify `batch_test_config.yaml` in `config` for your inference specifications. Run `batch_test_main.py` with `hydra`-style parameter passing to evaluate.

e.g.
```
python script_main.py ozername=SGD basic_learning_rate=5e-5
python batch_test_main.py dataset=path/to/your/data foldername=path/to/save/results
```

# Facial point cloud-based face verification
Use scripts in `FPCT-ID`. Put your data split record (simply in txt format) in `DataSplit`. Modify `config.yaml` in `config` for your training specifications. Run `script_main.py` with `hydra`-style parameter passing to train. Modify `batch_inter_test_config.yaml` in `config` for your inference specifications. Run `batch_inter_test_main.py` with `hydra`-style parameter passing. Run `GenROC.py` to get ROC and AUC using a pair of training data inference and test data inference to infer.

e.g.
```
python script_main.py ozername=SGD basic_learning_rate=5e-5
python batch_inter_test_main.py dataset=path/to/your/data foldername=path/to/save/results
python GenROC.py
```

# VisualCMT
Download and unzip `VisualCMT_compressed.zip`, then double click `VisualCMT.exe` to use. It is an interactive software for Coordinate-wise Monotonic Transformations (CMTs). The software takes in comma-separated value (CSV) formatted data with X, Y, and Z coordinates in the first three columns, and R, G, B color components in the last three columns. The software automatically normalizes the X, Y, and Z coordinates to the range of [-1, 1], while RGB values are expected to be provided in decimal magnitudes. Mouse drag and wheel spin are supported in the input and output view windows. Six CMTs are available including Coordinate-wise Rank Transformation (CRT).

**Input data structure example:**

![data example](/DataStructureExample.png)

**VisualCMT screenshot:**

![VisualCMT](/VisualCMT.png)

The software was packed using PyInstaller and mannually reduced volume in a Win11 system, please report issue if the software fail to run in your system.
