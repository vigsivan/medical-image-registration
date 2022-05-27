# Voxelmorph SOP

Vignesh Sivan

# Pre-requisites

- Python 3.6+
- The following packages

```bash
absl-py==0.15.0
astunparse==1.6.3
backcall
backports.functools-lru-cache
cached-property==1.5.2
cachetools==4.2.4
certifi==2021.10.8
charset-normalizer==2.0.10
click==8.0.4
cycler==0.11.0
debugpy
decorator
Deprecated==1.2.13
einops==0.4.0
entrypoints
flatbuffers==1.12
fonttools==4.28.5
gast==0.3.3
google-auth==2.3.3
google-auth-oauthlib==0.4.6
google-pasta==0.2.0
grpcio==1.32.0
h5py==2.10.0
humanize==4.0.0
idna==3.3
imageio==2.13.5
importlib-metadata==4.10.1
ipykernel
ipython
jedi
joblib==1.1.0
jupyter-client
jupyter-core
keras==2.7.0
Keras-Preprocessing==1.1.2
kiwisolver==1.3.2
libclang==12.0.0
Markdown==3.3.6
matplotlib==3.5.1
matplotlib-inline
monai==0.8.1
nest-asyncio
networkx==2.6.3
neurite
nibabel==3.2.1
numpy==1.19.5
oauthlib==3.1.1
opt-einsum==3.3.0
packaging==21.3
parso
pexpect
pickleshare
Pillow==9.0.0
prompt-toolkit
protobuf==3.19.3
ptyprocess
pyasn1==0.4.8
pyasn1-modules==0.2.8
Pygments
pyparsing==3.0.6
pystrum
python-dateutil
PyWavelets==1.2.0
pyzmq==19.0.2
requests==2.27.1
requests-oauthlib==1.3.0
rsa==4.8
scikit-image==0.19.1
scikit-learn==1.0.2
scipy==1.7.3
SimpleITK==2.1.1
six
tensorboard==2.7.0
tensorboard-data-server==0.6.1
tensorboard-plugin-wit==1.8.1
tensorflow==2.7.0
tensorflow-estimator==2.7.0
tensorflow-io-gcs-filesystem==0.23.1
termcolor==1.1.0
threadpoolctl==3.0.0
tifffile==2021.11.2
torch==1.10.2
torchio==0.18.74
tornado
tqdm==4.62.3
traitlets
typer==0.4.0
typing-extensions==3.7.4.3
urllib3==1.26.8
voxelmorph
wcwidth
Werkzeug==2.0.2
wrapt==1.12.1
zipp==3.7.0
```

Use pip install -r requirements.txt to install them

# Data Preparation

Voxelmorph can read nifti (.nii.gz/.nii) or npy (.npy) files. When running the scripts, it expects a list of all of the files in txt format. A quick way to generate this list, if all of the files are in the same directory is to run the command 

```bash
ls <DIRECTORY> > image_list.txt
```

When segmentation is available, it expects the segmentation to have part of the same name as the image. I found that storing the segmentation in a separate folder while having the same name as its corresponding image.

When running segmentation on image pairs, you need to create a text file where each line has the two images separated by a space. E.g. for registering corresponding CT and MRI images, your pairs.txt file might look something this:

```bash
# pairs.txt
CT_sub1.nii.gz MR_sub1.nii.gz
...
CT_sub100.nii.gz MR_sub100.nii.gz
```

# Running voxelmorph

The scripts are stored in the voxelmorph/tf/scripts directory.

```bash
# NOTE: when the atlas is not provided, scan-scan registration is performed
# NOTE: the prefix commands are used to tell the script the parent directory of the images

# When segmentation is not available:
python train.py --img-list ~/t1t2_train.txt --img-prefix ~/cropped_reshaped/ --atlas ~/atlas_com/PAM50_t1_crop.nii.gz --model-dir hypermorph_cropped

# When segmentation is available:
python train_semisupervised_seg.py --img-list ~/t1t2_train.txt --img-prefix ~/cropped_reshaped/ --seg-prefix ~/cropped_reshaped_seg/ --atlas ~/atlas_com/PAM50_t1_crop.nii.gz --labels ~/labels.npy  --dice-loss-weight 1 --grad-loss-weight 1 --model-dir models_cropped_atlas

# Registering corresponding pairs:
python train_semisupervised_pairwise.py --pairs ~/t1t2_train_pairs.txt --img-prefix ~/cropped_reshaped/ --seg-prefix ~/cropped_reshaped_seg/ --labels ~/labels.npy --model-dir ./pairwise_checkpoints_cropped_even_more_reg --dice-loss-weight 1 --grad-loss-weight 10
```

# Running on Compute Canada

Here is an example slurm file that use for running on compute canada. You will need to replace your own sponsor’s name where indicated. You will also need to modify the names of the folders that your are using. Also note the data is transfereed to the tmpdir as it is the faster file system. 

```bash
#!/bin/bash                                                                                                                                                                    
                                                                                                                                                                               
#SBATCH --gres=gpu:1                                                                                                                                                           
#SBATCH --mem=4G                                                                                                                                                               
#SBATCH --time=18:00:00                                                                                                                                                        
#SBATCH --account=<your_sponsor>                                                                                                                                                
                                                                                                                                                                                                                                                                                                                        
                                                                                                                                                                               
source ~/scratch/vox_env/bin/activate                                                                                                                                          
cp ~/scratch/t1t2_com_32x128x128_0to1.tar.gz $SLURM_TMPDIR                                                                                                                     
cp ~/scratch/t1t2_com_32x128x128_seg.tar.gz $SLURM_TMPDIR                                                                                                                      
                                                                                                                                                                               
cd $SLURM_TMPDIR                                                                                                                                                               
tar -xf t1t2_com_32x128x128_0to1.tar.gz                                                                                                                                        
tar -xf t1t2_com_32x128x128_seg.tar.gz                                                                                                                                         
                                                                                                                                                                               
cd ~/scratch/voxelmorph/scripts/tf/                                                                                                                                            
                                                                                                                                                                               
python train_semisupervised_seg_mi.py --img-list ~/scratch/t1t2_train.txt --img-prefix $SLURM_TMPDIR/t1t2_com_32x128x128_0to1/ --seg-prefix $SLURM_TMPDIR/t1t2_com_32x128x128_seg/ --labels ~/scratch/labels.npy --model-dir ./semisupervised_models_mi --gpu 0 --dice-loss-weight 1 --grad-loss-weight 0.1
```