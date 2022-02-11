# Survival_RMS
Github page for Predicting survival of rhabdomyosarcoma patients based on deep-learning of hematoxylin and eosin images manuscript

## Set Up Environment
Two different conda environments are required
- `rms1`: Running 'subtype', 'myod1' and 'tp53' tasks
- `rms2`: Running 'segment' and 'sursvival' tasks


```
conda env create -f environment1.yml
conda activate rms1
conda install pytorch=1.7.1 torchvision=0.8.2 torchaudio cudatoolkit=10.2 -c pytorch
```

```
conda env create -f environment2.yml
conda activate rms2
conda install pytorch=1.7.1 torchvision=0.8.2 torchaudio cudatoolkit=10.2 -c pytorch
```

## Repository Structure

Below are the src subdirectories in the repository: 

- `src/results/`: output files are stored at different subdirectories
- `src/images/`: each subdirectory contains image files (svs, tif or png) to be inferenced
- `src/model_weights/`: trained model weights are located at different subdirectories

Below are the main python files in the repository:

- `src/subtype_inference.py`: WSI subtype classification
- `src/segmentation_inference.py`: Pixel-level segmentation of different features in WSIs
- `src/tp53_inference.py`: TP53 mutation detection
- `src/myod1_inference.py`: MYOD1 mutation detection
- `src/survival_inference1.py and src/survival_inference2.py`: Risk prediction
