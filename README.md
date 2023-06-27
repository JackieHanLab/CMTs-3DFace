## _Privacy-Preserving Age Estimation from Human 3D Facial Data with Coordinate-Wise Monotonic Transformations_ 
### by Xinyu Bruce Yang, Jing-Dong Jackie Han _et al_.

![Face Transformer](/FaceTransformer.jpg)
> Created in NightCafe with prompts "a cyberpunk neural network as background with a 3D meshed construction toy-like face without body in the center and robotic arms in the periphery touching the face with flash and flares. Picture in no more than 6 colors with the "viridis" palette."

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

```
    x         y         z        xn        yn        zn         r         g         b
0 -0.784874 -0.244051 -0.536802 -0.455303 -0.903667 -0.458548  0.053385  0.033854  0.020833
1 -0.779600 -0.246530 -0.537601 -0.384419 -0.919449 -0.497628  0.057292  0.037760  0.023438
2 -0.795638 -0.237591 -0.536667 -0.584383 -0.838801 -0.435541  0.061198  0.031250  0.022135
3 -0.789381 -0.241684 -0.536477 -0.541382 -0.862164 -0.443775  0.052083  0.027344  0.020833
4 -0.787205 -0.242815 -0.536962 -0.496568 -0.885021 -0.457656  0.057292  0.033854  0.022135
```

**VisualCMT screenshot:**

![VisualCMT](/VisualCMT.png)

The software was packed using PyInstaller and mannually reduced volume in a Win11 system, please report issue if the software fail to run in your system.
