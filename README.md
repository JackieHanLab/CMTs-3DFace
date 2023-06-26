## Code for "_Privacy-Preserving Age Estimation from Human 3D Facial Data with Coordinate-Wise Monotonic Transformations_"

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
python batch_test_main.py dataset=path/to/your/data foldername=path/to/save/results
python GenROC.py
```
