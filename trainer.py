import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import datetime as dt
import time
import json
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, auc
from copy import deepcopy

# loss functions were moved to losses.py
from losses import *

class Trainer():
	def __init__(self, model, scheduler, config, globals):
		## training stuff
		self.model = model
		self.device = config['cuda_device']
		self.model.to(self.device)
		# automatic mixed precision is used for faster training
		self.amp = config['amp']
		
		self.batch_size = config['batch_size']
		self.iters_per_val = globals['iterations_per_validation']
		# linear learning rate warmup is used
		self.warmup_iters = config['warmup_iterations']
		
		# loss function
		self.loss = loss_mapping[config['loss']]
		config['loss_inner'] = loss_inner_mapping[config['loss']]
		
		# optimizer
		model_params = self.model.parameters()
		# from testing, ADAM > SGD
		if config['optimizer'] == 'sgd':
			model.optimizer = torch.optim.SGD(
								model_params,
								lr = config['start_lr'],
								momentum = 0.9,
								weight_decay = config['weight_decay'],
								nesterov = True)
		elif config['optimizer'] == 'adam':
			model.optimizer = torch.optim.AdamW(
								model_params, 
								lr = config['start_lr'],
								weight_decay = config['weight_decay']
								)
		elif config['optimizer'] == 'nadam':
			model.optimizer = torch.optim.NAdam(
								model_params, 
								lr = config['start_lr']
								)
		elif config['optimizer'] == 'radam':
			model.optimizer = torch.optim.RAdam(
								model_params, 
								lr = config['start_lr']
								)
		else:
			raise ValueError(config['optimizer'])
		
		## validation and performance stuff
		self.model.best = False # variable to keep track of whether to update best scores
		self.model.best_val_mean_AUC = 0
		self.scheduler = scheduler(config, globals)
		
		# eval function
		self.eval_fn = all_pairs_evaluation
		
		# cache triplet indices for validation
		self.val_triplets = None
		
		## other config stuff
		# angular margin stuff
		if 'cos' in config['loss'] or 'softmax' in config['loss']:
			try:
				config['arcface_margin'] = config['arcface_margin']
			except KeyError:
				config['arcface_margin'] = 0
			try:
				config['cosface_margin'] = config['cosface_margin']
			except KeyError:
				config['cosface_margin'] = 0
		
		if '_ce' in config['loss']:
			try:
				config['decoupled_softmax'] = config['decoupled_softmax']
			except KeyError:
				config['decoupled_softmax'] = False
		
		# temperature
		# controls the radius of the hypersphere which cosine losses project feature vectors onto
		try:
			config['temperature'] = config['temperature']
		except:
			if 'cos' in config['loss'] or 'softmax' in config['loss']:
				config['temperature'] = 10
			else:
				config['temperature'] = 1
		
		# replace classification relu with softplus?
		try:
			config['_soft'] = config['_soft']
		except KeyError:
			config['_soft'] = False
		
		# default margin
		try:
			config['margin'] = config['margin']
		except KeyError:
			config['margin'] = 0
		
		# ring loss
		# encourages representations to be distributed on a hypersphere
		# supposedly helps cases where small norm leads to drastic angle differences
		# unfortunately, it leads to training instability
		# TODO: has not been tested in a while, may need to be fixed
		try:
			if config['ring_loss']:
				self.ring_loss = True
				## try radius as a learnt parameter
				# self.ring_loss_radius = nn.Parameter(torch.tensor([self.model.feat_dim**0.5])).to(self.device)
				self.ring_loss_radius = nn.Parameter(torch.tensor([1.])).to(self.device)
			else:
				self.ring_loss = False
		except KeyError:
			self.ring_loss = False
		
		# center loss
		# basically distance loss on positives, only for cosine losses
		# also leads to training instability
		# TODO: has not been tested in a while, may need to be fixed
		try:
			self.center_loss = config['center_loss']
			if self.center_loss: 
				assert 'cos' in config['loss']
		except KeyError:
			self.center_loss = False
		
		## tiny optimizations for triplet construction
		# only CE losses require explicit negatives
		self._build_negatives =	('_ce' in config['loss']) or \
								('contrastive' in config['loss']) or \
								('triplet' in config['loss'])
		# only centroid losses require groups
		# centroid losses have been removed so you won't find them
		self._build_groups =	('centroid' in config['loss']) or \
								(self.center_loss)
		
		self.config = config
		self.globals = globals
	
	# this automates whole training process
	def train(self, train_loader, val_loader, test_loader, skip_to_iter=0):
		# loop stuff
		loop = True
		iter_per_epoch = len(train_loader)
		epoch = 0
		skip_to_iter += 1
		checkpoint = 1 + int(skip_to_iter / self.iters_per_val)
		
		do_warmup = (not self.globals['DEBUG']) and (skip_to_iter > 0)
		
		# automatic mixed-precision
		scaler = torch.cuda.amp.GradScaler()
		
		# performance tracking
		# there used to be extra contrastive objectives here, but they have been removed so total_loss_mean == total_loss_patient
		total_loss_mean = 0
		total_loss_patient = 0
		nan_strikes = 0
		
		# initialize warmup
		if do_warmup:
			for group in self.model.optimizer.param_groups:
				group['lr'] /= self.warmup_iters
		
		print('Training Begins...')
		training_start_time = time.time()
		while loop:
			epoch_start_time = time.time()
			for i, batch in enumerate(train_loader):
				iteration = epoch*iter_per_epoch + i + skip_to_iter
				## train iteration
				self.model.to(self.device)
				self.model.train()
				scaler._enabled = True
				with torch.enable_grad():
					self.model.optimizer.zero_grad()
					with torch.cuda.amp.autocast(self.amp):
						# total_images x D
						features = self.model(batch['images'])
						
						patient_triplets = self.build_training_triplets(batch['id2patient'])
						patient_triplets['idx'] = batch['idx']
						
						patient_features = features
						
						patient_loss = self.loss(
							patient_features,
							patient_triplets,
							self.config,
							model_extras=self.model.extras
						)
						loss = patient_loss
						
						if self.ring_loss:
							norms = features.norm(dim=-1)
							loss = loss + F.mse_loss(norms, self.ring_loss_radius.expand_as(norms))
						
						if self.center_loss:
							loss = loss + center_loss(features, patient_triplets)
					
					## for debugging nan losses...
					if loss.isnan() and self.amp:
						print('\nAMP nan loss! swapping to full-precision')
						scaler._enabled = False
						features = self.model(batch['images'])
						patient_features = features
						patient_loss = self.loss(
							patient_features,
							patient_triplets,
							self.config,
							model_extras=self.model.extras
						)
						loss = patient_loss
						
						if self.ring_loss:
							norms = features.norm(dim=-1)
							if self.ring_loss_radius is not None:
								## moving-average version
								self.ring_loss_radius = 0.001 * norms.detach().mean() + 0.999 * self.ring_loss_radius
								loss = loss + F.mse_loss(norms, self.ring_loss_radius.expand_as(norms))
							else:
								## moving-average version
								self.ring_loss_radius = norms.detach().mean()
								loss = loss + F.mse_loss(norms, self.ring_loss_radius.expand_as(norms))
					
					if loss.isnan():
						nan_strikes += 1
						if nan_strikes == 3:
							raise Exception('irrecoverable nan loss!')
					else:
						nan_strikes = 0
						total_loss_patient += patient_loss.item()
												
						# AMP update step
						scaler.scale(loss).backward()
						scaler.step(self.model.optimizer)
						scaler.update()
						
						total_loss_mean += loss.item()
				# print to terminal
				if (iteration > 0 and iteration % 100 == 0) or self.globals['DEBUG']:
					now = time.time()
					print('EP.', epoch, end='\t')
					print('{}%'.format(np.round(i/len(train_loader)*100,1)), end='\t')
					print('Loss: {:e}'.format(total_loss_mean/(((iteration-1)%self.iters_per_val)+1)), end='\t')
					# iterations per second
					print('i/s:', np.round((iteration+1-skip_to_iter)/(now-training_start_time),1), end='\t')
					# time per epoch
					print('t/e:', int((now-training_start_time)/((iteration+1-skip_to_iter)/len(train_loader))), 's', end='\t')
					# estimated time until validation
					print('etv:', int((self.iters_per_val-(iteration%self.iters_per_val))/((iteration+1-skip_to_iter)/(now-training_start_time))), 's',end='\t')
					# elapsed time
					print('t:', int(now-training_start_time), 's', end='\r')
				## validate
				if (iteration>0 and iteration%self.iters_per_val==0) or (self.globals['DEBUG'] and (iteration-skip_to_iter) == 20):
					print('')
					print('VAL {}'.format(checkpoint), end='\t')
					
					print('Iter {}\tEpoch {}-{}'.format(
						iteration,
						epoch,
						i
					))
					
					train_loss_mean		= total_loss_mean	/ self.iters_per_val
					train_loss_patient	= total_loss_patient/ self.iters_per_val
					
					print('Validating...')
					
					self.model.eval()
					with torch.no_grad():
						val_dict = self.test(val_loader, test=False)
					
					# check if model should continue training
					self.model, loop = self.scheduler.check(self.model, val_dict['patient-contrast_image_AUC'], checkpoint = checkpoint, iter = iteration)
					
					# update best scores if new best score
					if self.model.best:
						print('new best!')
						self.model.best_val_mean_AUC = val_dict['mean_AUC']
					print('')
					now = time.time()
					
					
					if self.globals['DEBUG'] or not loop:
						# quickly stop training
						print('Training Complete')
						print('Best val mean AUC: {}'.format(
							np.round(np.log(self.model.best_val_mean_AUC/(1-self.model.best_val_mean_AUC)),4)
						))
						print('')
						loop = False
						break
					
					# reset trackers
					total_loss_mean = 0
					total_loss_patient = 0
					checkpoint += 1
				## end validation
				
				# lr warmup
				# iteration goes up to (and including) self.warmup_iters-2
				# last multiplication cancels out original (/= self.warmup_iters)
				if iteration <= (self.warmup_iters-2) and do_warmup:
					factor = (iteration+2)/(iteration+1)
					for group in self.model.optimizer.param_groups:
						group['lr'] *= factor
			
			##### end for loop
			epoch += 1
			print('')
		
		##### end while loop
		
		# load best model
		if not self.globals['DEBUG']:
			checkpoint = torch.load('{}/best.pt'.format(self.config['_savedir']))
			self.model.load_state_dict(checkpoint['model_state_dict'])
			# rerun val set
			print('Revalidating...')
			val_dict = self.test(val_loader, test=False)
		
		
		# run test set
		print('Testing...')
		test_dict = self.test(test_loader, test=True)
		outputs = {}
		
		outputs['val_mean_AUC'] = val_dict['mean_AUC']
		outputs['test_mean_AUC'] = test_dict['mean_AUC']
		outputs['val_mean_AUC_patient'] = val_dict['patient-contrast_image_AUC']
		
		with open('{}/stats.json'.format(self.config['_savedir']),'w') as fp:
			json.dump(outputs,fp)
		
		return self.model
	
	def test(self, loader, test=False):
		# inference
		patient_features = []
		id2patient = []
		
		self.model.eval()
		self.model.to(self.device)
		print('\tbuilding features...', end='')
		with torch.no_grad():
			for batch in loader:
				features = self.model(batch['images'])
				patient_features.append(features.cpu().detach())
				if self.val_triplets is None or test:
					id2patient.append(batch['id2patient'])
			self.model.to('cpu', non_blocking=True)
			
			if self.val_triplets is None or test:
				# collect mappings
				# mappings are 0-indexed
				# [0,1,2] [0,1,2,3] etc.
				# target:
				# [0,1,2] [3,4,5,6]
				print('building triplets...', end='')
				current_offset = 0
				for i, thing in enumerate(id2patient):
					temp = thing[-1]+1
					id2patient[i] += current_offset
					current_offset += temp
				id2patient = torch.cat(id2patient)
				
				triplets = build_eval_triplets(id2patient)
				if self.val_triplets is None:
					self.val_triplets = triplets
					self.val_id2patient = id2patient
			else:
				triplets = self.val_triplets
				id2patient = self.val_id2patient
			print('done')
			
			
			outputs = {}
			
			AUCs = []
			print('\tevaluating patient-level...', end='\r')
			patient_features = torch.cat(patient_features)
			patient_metrics = self.eval_fn( 
				patient_features,	# A x D
				triplets,		# patients x P
				self.config
			)
			del patient_features
			for k,v in patient_metrics.items():
				outputs['patient-contrast_{}'.format(k)] = v
			AUCs.append(patient_metrics['image_AUC'])
			print('\t_________________________', end='\r')
			print('P_AUC:', np.round(patient_metrics['image_AUC'], 4), end='\t')
			print('P_AUC_logodds:',
				np.round(
					np.log(
						patient_metrics['image_AUC']/(1-patient_metrics['image_AUC'])
					)
				,4)
			)
		
		outputs['mean_AUC'] = np.mean(AUCs)
		
		self.model.to(self.device, non_blocking=True)
		print('')
		
		return outputs
	
	def rank_for_figures(self, loader):
		patient_features = []
		id2patient = []
		id2laterality = []
		files = []
		
		self.model.eval()
		self.model.to(self.device)
		print('\tbuilding features...', end='')
		with torch.no_grad():
			for batch in loader:
				features = self.model(batch['images'])
				patient_features.append(features.cpu().detach())
				id2patient.append(batch['id2patient'])
				id2laterality.append(batch['id2laterality'])
				files.append(batch['files'])
			self.model.to('cpu', non_blocking=True)
			
			# collect mappings
			# mappings are 0-indexed
			# [0,1,2] [0,1,2,3] etc.
			# target:
			# [0,1,2] [3,4,5,6]
			print('building triplets...', end='')
			current_offset = 0
			for i, thing in enumerate(id2patient):
				# assert thing[-1] == max(thing)
				temp = thing[-1]+1
				id2patient[i] += current_offset
				current_offset += temp
			id2patient = torch.cat(id2patient)
			id2laterality = torch.cat(id2laterality)
			
			triplets = build_eval_triplets(id2patient)
			print('done')
			
			print('\tevaluating patient-level...', end='\r')
			patient_features = torch.cat(patient_features)
			files = np.concatenate(files) # A
			# A x A
			img_scores, img_labels = all_pairs_scoring( 
				patient_features,	# A x D
				id2patient,		# A
				self.config
			)
		print('')
		self.model.to(self.device)
		A = len(img_scores)
		
		## calc ROC to get threshold
		# remove diagonals	(A-1) x (A-1)
		_img_scores = img_scores[~torch.eye(A).bool()]
		_img_labels = img_labels[~torch.eye(A).bool()].int()
		
		fpr, tpr, thres = roc_curve(_img_labels.flatten(), _img_scores.flatten())
		
		return img_scores, img_labels, files, id2laterality, (fpr,tpr,thres)
	
	def build_training_triplets(self,id2patient):	# 0 to N-1 inclusive
		if id2patient[0] is None:
			# current loss doesn't use triplets
			return {}
		outputs = {'id2patient' : id2patient}
		index = torch.arange(len(id2patient))
		# anchors
		patient_positives = []	# total(images) x var(positives)
		if self._build_negatives:
			patient_negatives = []	# total(images) x var(negatives)
			for i,patient in zip(index,id2patient):
				# POS:	SAME PATIENT
				# NEG:	DIFFERENT PATIENT
				p_pos = index[id2patient==patient]
				p_neg = index[id2patient!=patient]
				patient_positives.append(p_pos)
				patient_negatives.append(p_neg)
			# padding index is 0, but image IDs start from 0 as well, so temporarily pad with -1
			p_neg = nn.utils.rnn.pad_sequence(patient_negatives, batch_first=True, padding_value=-1)
			# revert initial -1 offset
			outputs['negatives'] = p_neg + 1
		else:
			for i,patient in zip(index,id2patient):
				p_pos = index[id2patient==patient]
				patient_positives.append(p_pos)
		p_pos = nn.utils.rnn.pad_sequence(patient_positives, batch_first=True, padding_value=-1)
		outputs['positives'] = p_pos + 1
		
		if self._build_groups:
			# groups
			patient_groups = []	# patients x var(positives)
			patient_count = id2patient[-1]+1
			for patient in np.arange(patient_count):
				## patient
				p_grp = index[id2patient==patient]
				patient_groups.append(p_grp)
			# padding index is 0, but image IDs start from 0 as well, so temporarily pad with -1
			p_grp = nn.utils.rnn.pad_sequence(patient_groups, batch_first=True, padding_value=-1)
			# revert initial -1 offset
			outputs['patient_groups'] = p_grp + 1
		
		return outputs


def build_eval_triplets(id2patient):	# 0 to N-1
	assert id2patient[-1] == max(id2patient)
	patient_count = id2patient[-1]+1
	patient_positives = []	# patients x var(positives)
	
	index = torch.arange(len(id2patient))
	for patient in np.arange(patient_count):
		## patient
		p_pos = index[id2patient==patient]
		patient_positives.append(p_pos)
	
	# pad arrays. padding index is 0 so init with -1
	p_pos = nn.utils.rnn.pad_sequence(patient_positives, batch_first=True, padding_value=-1)
	
	# revert initial -1 offset
	p_pos += 1	# 1 to N
	return p_pos


'''
INPUT
labels	P x K
scores	P x K
OUTPUT
tensor	K
'''
def _eval_topK_metrics(labels, scores, K=-1):
	if K == -1:
		K = len(scores[0])
	# P x 1
	max_poscount = labels.sum(dim=-1, keepdims=True)
	
	## get top K scores
	# filtering for top K first is faster than filtering after
	# P x K
	scores, idx = scores.topk(K, dim=-1)
	labels = labels[torch.arange(len(labels)).unsqueeze(-1), idx]
	
	## sort descending
	scores, idx = scores.sort(dim=-1, descending=True)
	labels = labels[torch.arange(len(labels)).unsqueeze(-1), idx]
	
	## cumsum
	# P x K
	poscount = labels.cumsum(dim=-1)
	'''
	1 0 1 1 0 0
	cumsum
	1 1 2 3 3 3
	
	rec@K = cumsum / P
	'''
	# (P,K) -> K
	recall_sum = (poscount / max_poscount).sum(dim=0)
	
	return recall_sum


# shared evaluation code to clean some code bloat
def _eval_final_calculations(
		valid_image_count,
		
		image_recall_at_sum,
		image_max_recall_at_sum,
		
		valid_patient_count,
		
		all_image_scores,
		all_image_labels,
		all_max_scores,
		all_agg_labels
	):
	outputs = {}
	
	# CMC (recall @ K, up to 100)
	outputs['image-mean_recall-at'] = image_recall_at_sum.cpu().numpy() / valid_image_count
	outputs['image-mean_max-score_recall-at'] = image_max_recall_at_sum.cpu().numpy() / valid_image_count
	
	## calculate global metrics
	all_image_scores = torch.cat(all_image_scores)
	all_image_labels = torch.cat(all_image_labels)
	
	all_max_scores = torch.cat(all_max_scores)
	all_agg_labels = torch.cat(all_agg_labels)
	
	# raw gallery
	outputs['image_ROC_fpr'], outputs['image_ROC_tpr'], _ = \
		roc_curve(all_image_labels, all_image_scores)
	outputs['image_AUC'] = auc(outputs['image_ROC_fpr'], outputs['image_ROC_tpr'])
	
	# aggregated gallery
	outputs['max-score_ROC_fpr'], outputs['max-score_ROC_tpr'], _ = \
		roc_curve(all_agg_labels, all_max_scores)
	outputs['max-score_AUC'] = auc(outputs['max-score_ROC_fpr'], outputs['max-score_ROC_tpr'])
	
	for i in range(10):
		temp = outputs['image_ROC_tpr'][outputs['image_ROC_fpr'] < 10**(-i)]
		outputs['image_sen_fpr_1e-{}'.format(i)] = temp[-1] if len(temp) > 0 else 0
	return outputs


def all_pairs_evaluation(
		features,	# A x D
		patient_idx,	# K x I
		config,
		**kwargs
	):
	device = config['cuda_device']
	A,D = features.shape
	K,I = patient_idx.shape
	
	do_cos = 	('cos' in config['loss']) or \
				('softmax' in config['loss'])
	if do_cos:
		features = F.normalize(features)
	# A		anchors (images)
	# K		patients
	# I		items (0 padded)
	
	# doing in 1 operation takes up waaay too much memory
	# 500 x 224 x 5974 = 2.5GB for K x P x N float32 tensor
	# instead, iterate over K
	features = F.pad(features, (0,0,1,0)).to(device)
	
	# note: unlike training loss, diagonal values must be manually filtered here
	
	patient_idx = patient_idx.to(device)	# K x I
	patient_embeds = F.embedding(patient_idx, features) # K x I x D
	if do_cos:
		_patient_embeds = patient_embeds.reshape(K*I,D).transpose(0,1) # D x K*I
	else:
		_patient_embeds = patient_embeds.reshape(K*I,D) # K*I x D
	mask = patient_idx.bool()				# K x I
	
	## lists for collecting metrics
	# raw scores for global metrics
	all_image_scores = []		# K x var(P)*(P-1+N)
	all_image_labels = []		# K x var(P)*(P-1+N)
	
	all_max_scores = []			# K x var(P)*K
	all_agg_labels = []			# K x var(P)*K
	
	image_recall_at_sum = None
	image_max_recall_at_sum = None
	
	valid_image_count = valid_patient_count = 0
	# iterate over K patients
	for k, (pos_embeds, pos_mask) in enumerate(zip(patient_embeds, mask)):
		# k				constant
		# pos_embeds	I x D
		# pos_mask		I
		
		# P x D
		pos_embeds = pos_embeds[pos_mask]
		P = len(pos_embeds)
		
		neg_indexer = torch.arange(K, device=device)!=k
		
		if do_cos:
			# dot product on normalized vectors
			# (P,D) x (D,K*I) -> (P,K*I) -> (P,K,I)
			scores = (pos_embeds @ _patient_embeds).reshape(P,K,I)
		else:
			# we use NEGATIVE distance because for ROC and PR calculations:
			# small = negative, big = positive
			# (P,D) - (K*I,D) -> (P,K*I) -> (P,K,I)
			scores = -torch.cdist(pos_embeds, _patient_embeds).reshape(P,K,I)
		
		# P x K-1 x I
		neg_scores = scores[:,neg_indexer]
		
		# K-1 x I
		neg_mask = mask[neg_indexer]
		
		## max similarity aggregation
		# mask out padding
		# negative distance range: [-inf,0], so hopefully -1e4 will not interfere with max op
		max_neg_scores = neg_scores.masked_fill(~neg_mask, -1e4)
		# (P,K-1,I) -> (P,K-1)
		max_neg_scores = max_neg_scores.max(dim=-1)[0]
		
		## get non-padding values from negatives
		# (P,K-1,I) -> (P,N)
		neg_scores = neg_scores[:,neg_mask]
		if P <= 1:
			# only 1 sample, no positives
			# probe only, no gallery
			# include negative pairs only
			# P x N
			all_scores = neg_scores.cpu()
			# P x K-1
			max_scores = max_neg_scores.cpu()
			
			## generate labels
			# P x N
			labels = torch.zeros_like(all_scores)
			# P x K-1
			agg_labels = torch.zeros_like(max_scores)
		else:
			## extract positive patient
			pos_scores = scores[:,k,pos_mask] # P x P
			## remove diagonals
			# (P,P-1)
			pos_scores = pos_scores[~torch.eye(*pos_scores.shape, device=device).bool()].reshape(P,P-1)
			# P x P-1+N
			all_scores = torch.cat((pos_scores, neg_scores), dim=-1)
			
			max_pos_scores = pos_scores.max(dim=-1, keepdim=True)[0]
			# P x 1+K-1 = P x K
			max_scores = torch.cat((max_pos_scores, max_neg_scores), dim=-1)
			# defer cpu() until after recall calculations
			
			## generate labels
			# (P,P-1+N)
			labels = torch.zeros_like(all_scores)
			labels[:,:P-1] = 1
			# (P,K)
			agg_labels = torch.zeros_like(max_scores)
			agg_labels[:,0] = 1
			
			## collect counts for performance metrics
			# (100) of values [0,P(P-1)]
			if image_recall_at_sum is None:
				image_recall_at_sum = _eval_topK_metrics(labels, all_scores)
				image_max_recall_at_sum = _eval_topK_metrics(agg_labels, max_scores)
			else:
				image_recall_at_sum += _eval_topK_metrics(labels, all_scores)
				image_max_recall_at_sum += _eval_topK_metrics(agg_labels, max_scores)
			
			valid_image_count += P
			valid_patient_count += 1
			
		## collect scores and labels
		all_image_labels.append(labels.flatten().cpu())
		all_image_scores.append(all_scores.flatten().cpu())
		
		all_max_scores.append(max_scores.flatten().cpu())
		all_agg_labels.append(agg_labels.flatten().cpu())
	return _eval_final_calculations(
				valid_image_count,
				
				image_recall_at_sum,
				image_max_recall_at_sum,
				
				valid_patient_count,
				
				all_image_scores,
				all_image_labels,
				all_max_scores,
				all_agg_labels
			)

def all_pairs_scoring(
		features,	# A x D
		id2patient,	# A
		config,
		**kwargs
	):
	device = config['cuda_device']
	A,D = features.shape
	
	do_cos = 	('cos' in config['loss']) or \
				('softmax' in config['loss'])
	if do_cos:
		features = F.normalize(features)
		features_T = features.T
	
	# A x A
	labels = id2patient.reshape(1,-1) == id2patient.reshape(-1,1)
	labels = labels.float()
	
	## lists for collecting metrics
	# raw scores for global metrics
	all_image_scores = torch.zeros((A,A))
	# iterate over K patients
	for i,feature in enumerate(features):
		if do_cos:
			# dot product on normalized vectors
			# (D) x (D,A) -> (A)
			all_image_scores[i] = (feature @ features_T).cpu()
		else:
			# we use NEGATIVE distance because for ROC and PR calculations:
			# small = negative, big = positive
			# (1,D) - (A,D) -> (A)
			all_image_scores[i] = (-torch.cdist(feature.unsqueeze(0), features)).cpu()
	
	return all_image_scores, labels

loss_mapping = {
	'cos_bce':							all_pairs_BCE_loss,
	'cos_ce':							all_pairs_CE_loss,
	'cos_bce_thermostat':				all_pairs_BCE_loss,
	'cos_ce_thermostat':				all_pairs_CE_loss,
	'margin':							all_pairs_margin_loss,
	'margin_classifier':				all_pairs_margin_loss,
	'cos_margin':						all_pairs_margin_loss,
	'cos_margin_thermostat':			all_pairs_margin_loss,
	'triplet':							all_triplet_loss,
	'triplet_classifier':				all_triplet_loss,
	'cos_triplet':						all_triplet_loss,
	'cos_triplet_thermostat':			all_triplet_loss,
	'distance_classifier_bce':			all_pairs_BCE_loss,
	'distance_classifier_ce':			all_pairs_CE_loss,
	'legacy_distance_classifier_bce':	all_pairs_BCE_loss,
	'softmax':							softmax_loss,
	'cos_softmax':						softmax_loss
}
loss_inner_mapping = {
	'cos_bce':							CE_loss_inner_cosine,
	'cos_ce':							CE_loss_inner_cosine,
	'cos_bce_thermostat':				CE_loss_inner_cosine,
	'cos_ce_thermostat':				CE_loss_inner_cosine,
	'margin':							margin_loss_inner_distance,
	'margin_classifier':				margin_loss_inner_distance,
	'cos_margin':						margin_loss_inner_cosine,
	'cos_margin_thermostat':			margin_loss_inner_cosine,
	'triplet':							triplet_loss_inner_distance,
	'triplet_classifier':				triplet_loss_inner_distance,
	'cos_triplet':						triplet_loss_inner_cosine,
	'cos_triplet_thermostat':			triplet_loss_inner_cosine,
	'distance_classifier_bce':			CE_loss_inner_distance_classifier,
	'distance_classifier_ce':			CE_loss_inner_distance_classifier,
	'legacy_distance_classifier_bce':	CE_loss_inner_distance_classifier,
	'softmax':							None,
	'cos_softmax':						None
}