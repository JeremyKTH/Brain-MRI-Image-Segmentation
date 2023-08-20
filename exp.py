import argparse
import os
from tqdm import tqdm
import copy

import torch
from torch import optim

from models.model import UNET
import data
import params as P
import utils


# TODO: Test experiment logic.
# TODO: Implement UNET model variant using Hebbian modules.
# TODO: Implement unsupervised pre-training using Hebbian methods.
# TODO: Define experimental configurations.
# TODO: Test Hebbian modules.
# TODO: Add other datasets.
# TODO: Run final experiments.


# Function to compute Dice loss
def dice_loss(inputs, targets, smooth=1e-4):
	#comment out if your model contains a sigmoid or equivalent activation layer
	inputs = torch.sigmoid(inputs)
	
	#flatten label and prediction tensors
	inputs = inputs.view(-1)
	targets = targets.view(-1)
	
	intersection = (inputs * targets).sum()
	dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)

	return dice_loss

# This function contains the model training logic
def train(loader, model, optimizer):
	running_loss = 0.
	running_acc = 0.
	num_samples = 0

	for data, targets in tqdm(loader, ncols=80):
		data = data.to(device=P.DEVICE)
		targets = targets.float().unsqueeze(1).to(device=P.DEVICE)
		
		predictions = model(data)
		
		# Loss and accuracy
		bce = torch.nn.functional.binary_cross_entropy_with_logits(predictions, targets)
		dice = dice_loss(predictions, targets)
		loss = bce + dice
		acc = 1 - dice
		
		# Update running stats
		running_loss += loss.item()
		running_acc += acc.item()
		num_samples += data.size(0)

		# backward
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		
	return running_loss/num_samples, running_acc/num_samples

# This function contains the model evaluation logic
def eval(loader, model):
	running_loss = 0.
	running_acc = 0.
	num_samples = 0

	for data, targets in tqdm(loader, ncols=80):
		data = data.to(device=P.DEVICE)
		targets = targets.float().unsqueeze(1).to(device=P.DEVICE)
		
		predictions = model(data)
		
		# Loss and accuracy
		bce = torch.nn.functional.binary_cross_entropy_with_logits(predictions, targets)
		dice = dice_loss(predictions, targets)
		loss = bce + dice
		acc = 1 - dice
		
		# Update running stats
		running_loss += loss.item()
		running_acc += acc.item()
		num_samples += data.size(0)
		
	return running_loss/num_samples, running_acc/num_samples

# Experiment logic
def launch(config_name, restart=False):
		config = utils.retrieve(config_name)
		save_path = os.path.join('results', args.config_name.replace('.', os.sep))
		
		print("Loading dataset...")
		train_loader, eval_loader = data.dataset.get_loaders(batch_size=config.get('batch_size', 8))
		
		print("Loading model...")
		model = UNET(in_channels=3, out_channels=1)
		model = model.to(P.DEVICE)
		optimizer = optim.SGD(model.parameters(), lr=config.get('lr', 1e-4), momentum=config.get('momentum', 0.9), weight_decay=config.get('wdec', 5e-4), nesterov=True)
		scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=config.get('sched_milestones', []), gamma=config.get('sched_gamma', 1))
		
		
		results = {'train_loss': {}, 'train_acc': {}, 'eval_loss': {}, 'eval_acc': {}}
		start_epoch = 1
		best_epoch = 0
		num_epochs = config.get('num_epochs', 50)
		
		if not restart:
			checkpoint = utils.load_dict(os.path.join(save_path, 'checkpoint.pt'))
			model.load_state_dict(checkpoint['model'])
			optimizer.load_state_dict(checkpoint['optimizer'])
			scheduler.load_state_dict(checkpoint('scheduler'))
			start_epoch = checkpoint['epoch']
			best_epoch = checkpoint['best_epoch']
			results = checkpoint['results']
		
		for epoch in range(start_epoch, num_epochs + 1):
			print("EPOCH {}/{}".format(epoch, num_epochs))
			
			#Train
			print("Training...")
			train_loss, train_acc = train(train_loader, model, optimizer)
			print("Train results at epoch {}: loss {}, acc {}".format(epoch, train_loss, train_acc))
			results['train_loss'][epoch] = train_loss
			results['train_acc'][epoch] = train_acc
			
			# Eval
			print("Evaluating...")
			eval_loss, eval_acc = eval(eval_loader, model)
			print("Eval results at epoch {}: loss {}, acc {}".format(epoch, eval_loss, eval_acc))
			results['eval_loss'][epoch] = eval_loss
			results['eval_acc'][epoch] = eval_acc
			
			if eval_acc > results['eval_acc'].get(best_epoch, 0): best_epoch = epoch
			print("Best epoch so far {}".format(best_epoch))
			print("with eval results: loss {}, acc. {}".format(results['eval_loss'][best_epoch], results['eval_acc'][best_epoch]))
			
			# LR schedule
			scheduler.step()
			
			# Save results
			utils.update_csv(results, os.path.join(save_path, 'results.csv'))
			utils.save_dict({
				'epoch': epoch + 1,
				'best_epoch': best_epoch,
				'results': results,
				'model': model.state_dict(),
				'optimizer': optimizer.state_dict(),
				'scheduler': scheduler.state_dict(),
			}, os.path.join(save_path, 'checkpoint.pt'))
			if epoch == best_epoch:
				utils.save_dict(copy.deepcopy(model.state_dict()), os.path.join(save_path, 'model.pt'))


if __name__ == "__main__":
	# Parse command line arguments
	parser = argparse.ArgumentParser()
	parser.add_argument('--config', default=P.DEFAULT_CONFIG, help="The experiment configuration you want to run.")
	parser.add_argument('--device', default=P.DEVICE, choices=P.AVAILABLE_DEVICES, help="The device you want to use for the experiment.")
	parser.add_argument('--restart', action='store_true', default=P.DEFAULT_RESTART, help="Whether you want to restart the experiment from scratch, overwriting previous checkpoints in the save path.")
	args = parser.parse_args()

	# Override default params
	P.DEVICE = args.device
	
	launch(args.config, restart=args.restart)
	
	print("\nFinished!")
	
