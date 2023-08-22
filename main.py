print('Importing libraries...')

import os
import json
import argparse

import torch
import numpy as np
import pandas as pd

from torch.utils.data import DataLoader

from dataset import *
from trainer import Trainer
from models import ContrastiveWrapper

###########################
# ARGPARSE
parser = argparse.ArgumentParser()
parser.add_argument('config', type=str, 
					help='path to a hyperparameter configs json')
parser.add_argument('device', type=str, 
					help='cuda device to use')

args = parser.parse_args()
CONFIG_DIR = '/'.join(args.config.split('/')[:-1])
CONFIG_FILE = args.config.split('/')[-1]

if __name__ == '__main__':

###########################
# GLOBALS
	
	# load globals
	with open('globals.json','r') as fp:
		globals = json.load(fp)
	
	# load hyperparams
	# open baseline
	with open(globals['baseline_json'],'r') as fp:
		config = json.load(fp)
	
	# update baseline with new hyperparams
	with open('{}/{}'.format(CONFIG_DIR, CONFIG_FILE),'r') as fp:
		config.update(json.load(fp))
	
	if args.device == 'cpu':
		config['cuda_device'] = torch.device('cpu')
	else:
		config['cuda_device'] = torch.device('cuda:{}'.format(args.device))
	
	# set model stuff
	exec('from schedulers import {} as SCHEDULER'.format(config['scheduler']))
	
	# run save dir
	config['_savedir'] = '.'.join(args.config.split('.')[:-1])
	
	os.makedirs(config['_savedir'], exist_ok=True)
	
	# IMAGEFILES
	IMAGE_DIR = globals['image_dir']
	
	def set_random_seed(seed):
		np.random.seed(seed)
		torch.manual_seed(seed)
		torch.cuda.manual_seed_all(seed)
	
	print('using configuration:', CONFIG_FILE)
	if globals['DEBUG']:
		print('DEBUGGING ACTIVE')
		config['batch_size'] = 3
	
	# set seed for reproducibility
	set_random_seed(config['seed'])
	print('Initializing Trainer...')
	
	# dataframe splits
	# image_df defines all images to be used; excluded images are removed during preprocessing
	try:
		meta_df = pd.read_csv(config['meta_override'])
	except KeyError:
		meta_df = pd.read_csv(globals['image_df'])
	
	## EDIT HERE TO ADAPT TO DIFFERENT DATASET!
	# don't forget to change globals.json
	# define the train/val/test sets
	split_df = pd.read_csv(globals['split_df'])
	split_df = split_df[['dicom_id','split']]
	meta_df = meta_df.merge(split_df, on='dicom_id')
	meta_df['path'] = 'p' + meta_df['subject_id'].astype(str).str[:2] + '/p' + meta_df['subject_id'].astype(str) + '/s' + meta_df['study_id'].astype(str) + '/' + meta_df['dicom_id'] + '.jpg'
	'''
	they only need these columns
	path		path to the image (from IMAGE_DIR, so '{}/{}'.format(IMAGE_DIR, path))
	subject_id
	'''
	meta_df = meta_df[['path', 'subject_id', 'split']]
	
	meta_train =	meta_df[meta_df.split == 'train']
	meta_val =		meta_df[meta_df.split == 'validate']
	meta_test =		meta_df[meta_df.split == 'test']
	## OKAY YOU CAN STOP NOW
	
	
	# dataset init
	train_set = ContrastivePatientDS(
					meta_train,
					IMAGE_DIR,
					img_size=config['frontal_img_size'], 
					augment=config['augment_level'],
					augment_pairs=config['augment_pairs'],
					augment_singleton=config['augment_singleton'],
					img_limit = config['patient_image_limit'] if config['batch_size'] > 1 else 9999
					)
	val_set = ContrastivePatientDS(
					meta_val,
					IMAGE_DIR, 
					img_size=config['frontal_img_size'], 
					augment=False,
					augment_pairs=False,
					augment_singleton=False,
					img_limit = 9999
					)
	test_set = ContrastivePatientDS(
					meta_test,
					IMAGE_DIR, 
					img_size=config['frontal_img_size'], 
					augment=False,
					augment_pairs=False,
					augment_singleton=False,
					img_limit = 9999
					)
	collate_fn = patient_collate_flatten
	
	train_loader = DataLoader(
					train_set, 
					batch_size = config['batch_size'], 
					shuffle = True, 
					num_workers = 0 if globals['DEBUG'] else globals['dataloader_workers'], 
					drop_last = True, 
					pin_memory = True, 
					collate_fn = collate_fn
					)
	val_loader = DataLoader(
					val_set, 
					batch_size = 16, 
					shuffle = False, 
					num_workers = globals['dataloader_workers'], 
					drop_last = False, 
					pin_memory = True, 
					collate_fn = collate_fn
					)
	test_loader = DataLoader(test_set, 
					batch_size = 16, 
					shuffle = False, 
					num_workers = globals['dataloader_workers'], 
					drop_last = False, 
					pin_memory = True, 
					collate_fn = collate_fn
					)
	config['validations_per_epoch'] = int(len(train_loader) / globals['iterations_per_validation']) + 1
	
	model = ContrastiveWrapper(config)
	model = model.to(config['cuda_device'])
	
	trainer = Trainer(model, SCHEDULER, config, globals)
	
	## do whole training process
	model = trainer.train(train_loader, val_loader, test_loader)