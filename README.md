We explore bio-inspired training solutions based on the Hebbian principle
for deep learning applications in the context of image segmentation tasks.
## Setup Environment	

This tutorial shows how to setup a python environment with the exact library versions. The tools shown here are `asdf` and `virtualenv`

### Install asdf

Install asdf using git

	git clone https://github.com/asdf-vm/asdf.git ~/.asdf --branch v0.12.0`

Then add to your .bashrc or .zshrc file

	. "$HOME/.asdf/asdf.sh"
	. "$HOME/.asdf/completions/asdf.bash"

For the changes to have an effect restart the shell.
### Install Python
Then install `python 3.11.4`

	asdf plugin add python
	asdf install python 3.11.4

Add to `.tool-versions`

	echo "python 3.11.4" >> .tool-versions

### Create virtualenv
Upgrade `pip` and install `virtualenv`

	pip install pip --upgrade
	pip install virtualenv

Create new virtual environment

	virtualenv venv

Activate the environment

	source venv/bin/activate

Install libraries

	pip install -r requirements.txt

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
Use `requirements.txt`
<!--
- Python  3.6
- PyTorch 1.8.1
-->
## Contacts
Gabriele Lagani: gabriele.lagani@phd.unipi.it
