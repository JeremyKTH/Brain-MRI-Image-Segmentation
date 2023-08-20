We explore bio-inspired training solutions based on the Hebbian principle
for deep learning applications in the context of image segmentation tasks.

## Usage
Launch experiment with:
```
python exp.py --config <config> --device <device> --restart
```
Where:
 - `<config>` is the name of a configuration dictionary, with dotted 
 notation, defined anywhere in your code. For example
 `configs.base.config_base`.
 - `<device>` can be `cpu`, `cuda:0`, or any device you wish to use for
 the experiment.
 - The flag `--restart` is optional. If you remove it, you can resume a 
 previously suspended experiment from a checkpoint, if available.

## Datasets
[Brain MRI images](https://www.kaggle.com/mateuszbuda/lgg-mri-segmentation)

## Requirements
- Python  3.6
- PyTorch 1.8.1

# Contacts
Gabriele Lagani: gabriele.lagani@phd.unipi.it
