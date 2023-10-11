# ADLM_SS2023_mmcl

TUM SS2023 - Applied Deep Learning in Medicine

<!-- ![Poster]()  -->

## Setup
### CONDA 

First run
`/opt/anaconda3/bin/conda init`

#### Create Env and Install Libs via CONDA :

```
conda create --name morph_1 --clone base
conda activate morph_1
```
Pytorch-CPU:
`conda install -c pytorch pytorch torchvision torchaudio cpuonly `

Pytorch-GPU[Cuda11.7]:
`conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia`

Pytorch-GPU[Cuda11.6]:
`conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia`

```
conda install -c pytorch captum -y
conda install -c conda-forge opencv -y 
conda install transformers tensorboardX pandas -y
```

Create the environment from the environment.yml file: 

`conda env create -f environment.yml`

Update environment from the environment.yml file: 

`conda env update -n my_env --file ENV.yaml`


### PIP
#### Create Env and Install Libs via PIP :

```
python3 -m venv morph_1
source morph_1/bin/activate
pip3 install torch torchvision torchaudio
pip install transformers tensorboardX pandas pycocotools captum  opencv-python
```

### Using Colab and Github

1. create a Github folder in your drive
2. Create a GitHub Personal access token 

  login to your account> Go to Settings> Developer settings>Click on Personal access tokens (classic)> Check the repo checkmark  
  
  Copy Personal access tokens [Don't share it]
  
3. Using  colab terminal, set an env varaibled with your Personal access token  `export GIT_TOKEN=<Personal access>`
4. Clone the repo via `git clone <url>`

### Cloning the repository from gitlab

#### Setup the SSH-Key
`ssh-keygen -t ed25519`
Set a password.
The key now lies under ~/.ssh
Type `cat _keyname_.pub`
Copy the result and enter it in gitlab under Profile -> Preferences -> SSH keys
Paste the key and save the result.

#### Clone the repository
on the server navigate to space where you want to clone the repostiory.
run Clone the repo via `git clone <url>`
copy the url form the gitlab clone via ssh

## Create a new Enviroment 
1. Create a folder in your drive to host all your virtual enviroments Ex `mkdir /content/drive/MyDrive/YehiaEnv`
2. Create a new env <Env_name> in  envs folder by command `cd /content/drive/MyDrive/YehiaEnv && virtualenv morph_1` 
3. Activate this env run `source /content/drive/MyDrive/YehiaEnv/morph_1/bin/activate`

##  Run the project : 

### Folder structure

1. data_scripts: contains the dataloader and augmentation functions
2. notebooks: contains notebooks for testing, debugging and developing features in the project. Also contains the scripts used for preparing the data (like cropping and resampling)
3. radiomics: all scripts/notebooks and files relating to the radiomics analysis
4. results:  place to store results of the experiments (against val set)
5. evaluate: store final evaluation of the experiments (against test set)
6. src: all code relating to the structured execution of the experiments
    1. models: define models and functions used in training like architectures and augmentation functions
    2. experiment.py: contains the training loop, runs one experiment from start to finish
    3. experiment_enums.py: define different experiments through the hyperparameters
    4. get_data.py: helper function to return the correct dataset depending on the hyperparameters
    5. logger.py: logging functionality for experiments
    6. main.py: loads experiments from the enums and executes them one after the other

### Usage
1. Go into models and define: the architectures and augmentations you want to use in your experiment
2. set the parameters in the experiment_enums to define your experiment
3. run the file main.py 
4. the experiment will run, the current status will be printed on the console and final results are saved under results

