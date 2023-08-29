import os
import csv
from importlib import import_module

import torch
from torch import nn


# Return formatted string with time information
def format_time(seconds):
	seconds = int(seconds)
	minutes, seconds = divmod(seconds, 60)
	hours, minutes = divmod(minutes, 60)
	return str(hours) + "h " + str(minutes) + "m " + str(seconds) + "s"


# Transforms shape tuple to size by multiplying the shape values
def shape2size(shape):
	size = 1
	for s in shape:
		size *= s
	return size


# Save data to csv file
def update_csv(results, path):
	os.makedirs(os.path.dirname(path), exist_ok=True)
	with open(path, mode='w', newline='') as csv_file:
		writer = csv.writer(csv_file)
		for name, entries in results.items():
			writer.writerow([name + '_epoch'] + list(entries.keys()))
			writer.writerow([name] + list(entries.values()))


# Save state dictionary file to specified path
def save_dict(state_dict, path):
	os.makedirs(os.path.dirname(path), exist_ok=True)
	torch.save(state_dict, path)


# Load state dictionary file from specified path
def load_dict(path):
	torch.load(path, map_location='cpu')


# Retrieve a custom module or object provided by the user by full name in dot notation as string. If the object is a
# dictionary, it is possible to retrieve a specific element of the dictionary with the square bracket indexing notation.
# NB: dictionary keys must always be strings.
def retrieve(name):
	if name is None:
		return None
	
	if '[' in name:
		name, key = name.split('[', 1)
		key = key.rsplit(']', 1)[0]
		prefix, suffix = name.rsplit('.', 1)
		return getattr(import_module(prefix), suffix)[key]
	
	prefix, suffix = name.rsplit('.', 1)
	return getattr(import_module(prefix), suffix)


# Neural network weight initialization
def init_weights(model):
	for m in model.modules():
		if isinstance(m, nn.Conv3d):
			# n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
			# m.weight.data.normal_(0, math.sqrt(2. / n))
			nn.init.kaiming_normal_(m.weight)
		elif isinstance(m, nn.BatchNorm3d):
			m.weight.data.fill_(1)
			m.bias.data.zero_()


# Count neural network parameters
def count_params(model):
	return sum(p.numel() for p in model.parameters())
