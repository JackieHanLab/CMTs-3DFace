Code for "Privacy-Preserving Age Estimation from Human 3D Facial Data with Coordinate-Wise Monotonic Transformations"

# Facial point cloud-based age estimation
Use scripts in **FPCT-age**. Put your data split record (e.g. in txt format) in **DataSplit**. Modify **config.yaml** in **config** for your training specifications. Run **script_main.py** with hydra-style parameter passing (https://hydra.cc/docs/intro/) to train. Modify **batch_test_config.yaml** for your inference specifications. Run batch_test_main.py with hydra-style parameter passing.
e.g.
'''
python script_main.py ozername=SGD basic_learning_rate=5e-5
python batch_test_main.py dataset=path/to/your/data foldername=path/to/save/results
'''

# Facial point cloud-based face verification
Use scripts in **FPCT-ID**. Put your data split record (e.g. in txt format) in **DataSplit**. Modify **config.yaml** in **config** for your training specifications. Run **script_main.py** with hydra-style parameter passing (https://hydra.cc/docs/intro/) to train. Modify **batch_test_config.yaml** for your inference specifications. Run batch_test_main.py with hydra-style parameter passing.
e.g.
'''
python script_main.py ozername=SGD basic_learning_rate=5e-5
python batch_test_main.py dataset=path/to/your/data foldername=path/to/save/results
'''

The folder "FPCT-ID" contains necessary code for training and evaluation of facial point cloud-based face verification model. 

The folder "Preprocessing" contains 

"FPCT-ID" and "preprocessing" 
