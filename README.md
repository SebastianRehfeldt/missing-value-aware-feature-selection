# Learning from Incomplete Data

Missing values can be frequently observed in datasets and hinder many approaches from being applied directly. 
This repository collects different approaches for feature selection and classification on these datasets and further comes with utilities to create synthetic datasets and simulate different missing mechanisms.

It was developed during my Master's thesis and has the aim to check whether the traditional approaches - imputation or deletion - are inferior to approaches which handle missing values internally.
Further, the repository includes a version of the RaR algorithm which was extended by handling missing values internally and by introducing active sampling of subspaces rather than random subsampling.

# Datasets

Datasets can be downloaded directly either from UCI (as csv) or from openml (as arff). To add a dataset on your own create a data folder at top level and inside a csv or arff folder if not already existent. Create a folder for your own dataset which name equals the name you pass to the dataloader. For arff datasets you need to name the file containing the dataset "data.arff". In case of csv files, the file must be called "data.csv". When loading a csv dataset for the first time, a file called "meta_data.json" will be created. It stores the feature names and types which are needed during the algorithms (can be edited).

# Evaluation

To try and test the algorithms, there is a jupyter script at the top-level which is called test.py (can be executed in vscode or atom using jupyter and hydrogen extension, can also be run as a script but without breakpoints). 

It shows how to load data, introduce missing values and how to run the developed algorithm on it.

# Installation (Windows)

Required installations
- Python 3.6 and pip (other versions might also work but are not tested)
- Microsoft Visual C++ Build Tools
    - https://visualstudio.microsoft.com/de/vs/community/
    - Tools für Visual Studio 2017 > Download Build Tools für Visual Studio 2017 (including Windows SDK)
- Gurobi optimizer (https://www.gurobi.com/downloads/gurobi-optimizer)
    - Get licence, install and run "grbgetkey \<key\>" in run and start menu (win+r)
    - Run "python setup.py install" in gurobi folder

Steps:
- Install requirements using pip (virtual environment is recommended)
    - Using git bash under Windows:
    - python3 -m venv rar
    - source rar/Scripts/activate
- pip install -r requirements.txt
    - (eventually ecos module does not succeed to install, install then using whl file from https://www.lfd.uci.edu/~gohlke/pythonlibs/)
- python setup.py build_ext --inplace

- (Jupyter when using venv and vs code extension):
    - Jupyter must be installed globally
    - Create kernel in venv: ipython kernel install --name=rar
    - Go to python site-packages and copy gurobipy folder to site-packages in venv
    - Test kernel from vs code

- Add root folder to system path or set workspace root in vs code

# Experiments and Statistics

The experiment folder contains scripts to compare different feature selection/ classification algorithms on synthetic or real-world datasets.
The data folder also contains statistics about the datasets and might include visualizations of datasets. If not present,the scripts to create these statistics and visualizations can be found in the scripts folder.