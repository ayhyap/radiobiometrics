print('Importing libraries...')

import os
import json
import argparse

import torch
import numpy as np
import pandas as pd
import pickle
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
	exec(config['scheduler_import'] + ' as SCHEDULER')
	
	# run save dir
	config['_savedir'] = '.'.join(args.config.split('.')[:-1])
	
	os.makedirs(config['_savedir'], exist_ok=True)
	
	# IMAGEFILES
	IMAGE_DIR = globals['image_dir']
	
	print('using configuration:', CONFIG_FILE)
	print('Initializing Trainer...')
	
	# dataframe splits
	# image_df defines all images to be used; excluded images are removed during preprocessing
	try:
		meta_df = pd.read_csv(config['image_df_override'])
	except KeyError:
		meta_df = pd.read_csv(globals['image_df'])
	
	## EDIT HERE TO ADAPT TO DIFFERENT DATASET!
	# don't forget to change globals.json
	# define the train/val/test sets
	meta_df['path'] = 'p' + meta_df['subject_id'].astype(str).str[:2] + '=p' + meta_df['subject_id'].astype(str) + '=s' + meta_df['study_id'].astype(str) + '=' + meta_df['dicom_id'] + '.jpg'
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
	val_set = ContrastivePatientDS(
					meta_val,
					IMAGE_DIR, 
					img_size=config['frontal_img_size'], 
					augment=False,
					augment_pairs=False,
					augment_singleton=False,
					img_limit=9999)
	test_set = ContrastivePatientDS(
					meta_test,
					IMAGE_DIR, 
					img_size=config['frontal_img_size'], 
					augment=False,
					augment_pairs=False,
					augment_singleton=False,
					img_limit=9999)
	collate_fn = patient_collate_flatten
	
	val_loader = DataLoader(val_set, 
						batch_size = 16, 
						shuffle = False, 
						num_workers = globals['dataloader_workers'], 
						drop_last = False, 
						pin_memory = True, 
						collate_fn = collate_fn)
	test_loader = DataLoader(test_set, 
						batch_size = 16, 
						shuffle = False, 
						num_workers = globals['dataloader_workers'], 
						drop_last = False, 
						pin_memory = True, 
						collate_fn = collate_fn)
	
	model = ContrastiveWrapper(config)
	model = model.to(config['cuda_device'])
	
	# load best model
	checkpoint = torch.load('{}/best.pt'.format(config['_savedir']), map_location=config['cuda_device'])
	model.load_state_dict(checkpoint['model_state_dict'])
	
	trainer = Trainer(model, SCHEDULER, config, globals)
	
	# rerun val set
	print('Revalidating...')
	val_dict = trainer.test(val_loader, test=True)
	with open('{}/val-results.pkl'.format(config['_savedir']),'wb') as fp:
		pickle.dump(val_dict, fp)
	
	# run test set
	print('Testing...')
	test_dict = trainer.test(test_loader, test=True)
	with open('{}/test-results.pkl'.format(config['_savedir']),'wb') as fp:
		pickle.dump(test_dict, fp)