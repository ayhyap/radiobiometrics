import torch
import torch.nn as nn
import torch.nn.functional as F

# treat each patient as a class and do softmax CE
def softmax_loss(
		features,	# A x D
		triplet_dict,
		config,
		**kwargs
	):
	balancing = config['patient_balancing']
	device = features.device
	
	# A
	labels = triplet_dict['idx'].to(device)
	
	# A x K
	logits = kwargs['model_extras']['classifier'](features)
	_arange = torch.arange(len(features), device=device)
	
	
	if config['arcface_margin']:
		logits[_arange,labels] = torch.cos(torch.acos(logits[_arange,labels]) + config['arcface_margin'])
	
	if config['cosface_margin']:
		logits[_arange,labels] = logits[_arange,labels] - config['cosface_margin']
	
	logits = logits * config['temperature']
	if config['hardneg_mining']:
		# ignore negatives with < 0 cosine similarity
		temp = logits[_arange,labels]
		logits[logits < 0] = -1e4
		logits[_arange,labels] = temp
	
	# rescale losses so each subject has equal contribution to loss regardless of number of images they have
	if balancing == 0:
		# use sum reduction then divide by nonzero count for mean
		loss = F.cross_entropy(logits, labels, reduction='mean')
	else:
		'''
		counts: [3 1 2]
		1
		1
		1
		1
		1
		1
		div counts (3)
		1/3
		1/3
		1/3
		1
		1/2
		1/2
		sum
		3
		div patients
		1
		'''
		# A
		loss = F.cross_entropy(logits, labels, reduction='none')
		# A
		counts = triplet_dict['positives'].bool().to(device).sum(-1)
		anchor_patientcount = triplet_dict['id2patient'][-1].item()+1
		loss = (loss / counts).sum() / anchor_patientcount
	return loss


def _cross_entropy(
		preds,	# N x K
		labels, # N
		decoupled=False,
		reduction='mean',
		**kwargs
	):
	if decoupled:
		# do softmax WITHOUT positive in denominator
		one_hot = F.one_hot(labels, num_classes = preds.shape[-1]).bool().to(preds.device)
		numer = -preds[one_hot]
		denom = torch.logsumexp(preds[~one_hot].reshape(len(preds),-1), dim=-1)
		# N
		loss = numer + denom
		if reduction == 'mean':
			loss = loss.mean()
		elif reduction == 'sum':
			loss = loss.sum()
		return loss
	else:
		return F.cross_entropy(preds, labels, reduction=reduction, **kwargs)

# calculates cross-entropy loss for all valid pairs
def all_pairs_CE_loss( 
		features,	# A x D
		triplet_dict,
		config,
		**kwargs
	):
	balancing = config['patient_balancing']
	device = features.device
	
	# A x ?
	positives = triplet_dict['positives'].to(device)
	negatives = triplet_dict['negatives'].to(device)
	pos_mask = positives.bool()
	neg_mask = negatives.bool()
	
	A,P = positives.shape
	_,N = negatives.shape
	
	# 1 x A
	id2patient = triplet_dict['id2patient'].unsqueeze(0).to(device)
	
	# A x A
	labels = (id2patient == id2patient.T).float()
	scores = config['loss_inner'](features, labels, config, **kwargs)
	
	# concat padding for indexing
	# A x 1+A
	scores = F.pad(scores, (1,0,0,0))
	
	temp = torch.arange(len(features), device=device).unsqueeze(-1)
	# A x ?
	pos = scores[temp,positives]
	neg = scores[temp,negatives]
	
	# mask out padding values so they don't contribute to loss
	# masking with inf or -inf makes nan loss, so mask with a big number instead
	pos = pos.masked_fill(~pos_mask, 1e4)
	neg = neg.masked_fill(~neg_mask, -1e4)
	
	# (A,P) -> (A,P,1)
	# (A,N) -> (A,P,N)
	pos = pos.unsqueeze(-1)
	neg = neg.unsqueeze(-2).expand(-1,P,-1)
	
	# (A,P,1+N)
	# positive labels are 0-th entry along last dimension
	out = torch.cat((pos,neg), dim=-1)
	# (A*P,1+N)
	out = out.reshape(A*P, 1+N)
	
	# positive labels are 0-th entry along last dimension
	# (A*P)
	labels = torch.zeros(len(out), dtype=int, device=device)
	
	# rescale losses so each subject has equal contribution to loss regardless of number of images they have
	if balancing == 0:
		# use sum reduction then divide by nonzero count for mean
		# loss = F.cross_entropy(out, labels, reduction='sum')
		loss = _cross_entropy(out, labels, decoupled=config['decoupled_softmax'], reduction='sum')
		loss = loss	/ pos_mask.sum()
	else:
		# loss = F.cross_entropy(out, labels, reduction='none')
		loss = _cross_entropy(out, labels, decoupled=config['decoupled_softmax'], reduction='none')
		# A x P
		loss = loss.reshape(A,P)
		# A
		counts = pos_mask.sum(-1)
		if balancing == 1:
			loss = (loss.sum(-1) / counts).mean()
		elif balancing == 2:
			anchor_patientcount = triplet_dict['id2patient'][-1].item()+1
			loss = (loss.sum(-1) / (counts*counts)).sum() / anchor_patientcount
		else:
			raise NotImplementedError(balancing)
	return loss


# calculates binary cross-entropy loss for all pairs
def all_pairs_BCE_loss(
		features,	# A x D
		triplet_dict,
		config,
		**kwargs
	):
	balancing = config['patient_balancing']
	device = features.device
	
	# 1 x A
	id2patient = triplet_dict['id2patient'].unsqueeze(0).to(device)
	
	# A x A
	labels = (id2patient == id2patient.T).float()
	scores = config['loss_inner'](features, labels, config, **kwargs)
	
	# A x A
	loss = F.binary_cross_entropy_with_logits(scores, labels, reduction='none')
	
	# rescale losses so each subject has equal contribution to loss regardless of number of images they have
	if balancing:
		# average each anchor's losses
		# A
		loss = loss.mean(-1)
		# average each patient's anchors
		# A
		anchor_poscounts = triplet_dict['positives'].to(device).bool().sum(-1)
		loss = loss / anchor_poscounts
		# average over all patients
		patient_count = triplet_dict['id2patient'][-1].item()+1
		loss = loss.sum() / patient_count
	else:
		loss = F.binary_cross_entropy_with_logits(scores, labels)
	
	return loss

# for use in CE/BCE loss
def CE_loss_inner_distance_classifier(
		features,	# A x D
		labels,		# A x A
		config,
		model_extras=None,
		**kwargs
	):
	# A x A
	distances = torch.cdist(features, features)
	# A x A
	classifier = model_extras['classifier']
	distances = classifier(distances.reshape(-1,1)).reshape_as(labels)
	
	if config['margin']:
		distances = distances - (1-labels)*config['margin']*classifier.weight.detach()
	
	# NEGATE! distance -> similarity
	return -distances

# for use in CE/BCE loss
def CE_loss_inner_cosine(
		features,	# A x D
		labels,		# A x A
		config,
		model_extras=None,
		**kwargs
	):
	scores = F.cosine_similarity(	features.unsqueeze(0),
									features.unsqueeze(1), dim=-1)
	
	if config['arcface_margin']:
		scores = torch.cos(torch.acos(scores) + config['arcface_margin'])
	
	if config['cosface_margin']:
		scores = scores - labels*config['cosface_margin']
	
	if model_extras is None:
		scores = scores * config['temperature']
	else:
		scores = model_extras['classifier'](scores.reshape(-1,1)).reshape_as(labels)
	
	return scores

# do margin loss for each probe-gallery pair distance
def all_pairs_margin_loss( 
		features,	# A x D
		triplet_dict,
		config,
		**kwargs
	):
	balancing = config['patient_balancing']
	device = features.device
	
	# 1 x A
	id2patient = triplet_dict['id2patient'].unsqueeze(0).to(device)
	
	# A x ?
	labels = (id2patient == id2patient.T).float()
	loss = config['loss_inner'](features, labels, config, **kwargs)
	
	# rescale losses so each subject has equal contribution to loss regardless of number of images they have
	# use sum then divide by nonzero count for mean
	if balancing == 0:
		# (???)		total valid pairs
		if config['hardneg_mining']:
			counts = (loss > 0).sum()
			loss = loss.sum() / counts.clamp(min=1)
		else:
			loss = loss.mean()
	else:
		if config['hardneg_mining']:
			counts = (loss > 0).sum(-1)
			loss = loss.sum(-1) / counts.clamp(min=1)
		else:
			loss = loss.mean(-1)
		pos_counts = triplet_dict['positives'].to(device).bool().sum(-1)
		loss = loss / pos_counts
		anchor_patientcount = triplet_dict['id2patient'][-1].item()+1
		loss = loss.sum() / anchor_patientcount
	return loss


# uses distances between pairs of features as hinge loss
# for use in all_pairs_margin_loss
def margin_loss_inner_distance(
		features,	# A x D
		labels,		# A x A
		config,
		model_extras=None,
		**kwargs
	):
	# A x A
	distances = torch.cdist(features, features)
	
	if model_extras is not None:
		# batch-normalized, weighted distances
		bn = model_extras['classifier']
		distances = bn(distances.reshape(-1,1)).reshape_as(labels)
		temperature = bn.weight.detach()
	else:
		temperature = config['temperature']
		distances = distances * temperature
	
	## negate negative pairs
	# mult negmask [-1,1]
	# pos: d(a,b)
	# neg:-d(a,b)
	loss = distances * (labels*2-1)
	
	## apply margin on negative pairs
	# pos: d(a,b)
	# neg: M-d(a,b)
	loss = loss + (config['margin']*(1-labels))*temperature
	
	## relu
	# pos: relu(d(a,b)) = d(a,b)
	# neg: relu(M-d(a,b))
	if config['_soft']:
		loss = F.softplus(loss)
	else:
		loss = torch.relu(loss)
	
	return loss

# uses cosine similarity between pairs of features as hinge loss
# for use in all_pairs_margin_loss
def margin_loss_inner_cosine(
		features,	# A x D
		labels,		# A x A
		config,
		model_extras=None,
		**kwargs
	):
	scores = F.cosine_similarity(	features.unsqueeze(0),
									features.unsqueeze(1), dim=-1)
	
	## negate positive pairs
	# mult posmask [1,-1]
	# pos:-s(a,b)
	# neg: s(a,b)
	loss = scores * (-labels*2 + config['margin'])
	
	# pos: 1-s(a,b)
	# neg: 0+s(a,b)
	loss = loss + labels
	
	## relu
	# pos: relu(1-s(a,b)) = 1-s(a,b)
	# neg: relu(s(a,b))
	if config['_soft']:
		loss = F.softplus(loss)
	else:
		loss = torch.relu(loss)
	
	## temperature
	if model_extras is None:
		loss = loss * config['temperature']
	else:
		scores = model_extras['classifier'](scores.reshape(-1,1)).reshape_as(labels)
	
	return loss

def all_triplet_loss( 
		features,	# A x D
		triplet_dict,
		config,
		**kwargs
	):
	balancing = config['patient_balancing']
	device = features.device
	
	# A x ?
	positives = triplet_dict['positives'].to(device)
	negatives = triplet_dict['negatives'].to(device)
	pos_mask = positives.bool()
	neg_mask = negatives.bool()
	
	A,P = positives.shape
	_,N = negatives.shape
	
	distances, temperature = config['loss_inner'](features, config, **kwargs)
	
	distances = F.pad(distances, (1,0,0,0))
	temp = torch.arange(len(distances), device=device).unsqueeze(-1)
	# A x ?
	pos = distances[temp,positives]
	neg = distances[temp,negatives]
	
	# (A,P) -> (A,P,1)
	# (A,N) -> (A,1,N)
	pos = pos.unsqueeze(-1)
	neg = neg.unsqueeze(-2)
	
	# A x P x N
	scores = pos - neg + config['margin']*temperature
	
	if config['_soft']:
		loss = F.softplus(scores)
	else:
		loss = torch.relu(scores)
	
	# A x P x N
	mask = pos_mask.unsqueeze(-1) * neg_mask.unsqueeze(1)
	
	# mask out padding values so they don't contribute to loss
	loss = loss.masked_fill(~mask, 0)
	
	if config['hardneg_mining']:
		mask = mask & (loss > 0)
	
	# rescale losses so each subject has equal contribution to loss regardless of number of images they have
	# use sum then divide by nonzero count for mean
	if balancing == 0:
		loss = loss[mask].sum() / mask.sum().clamp(min=1)
	else:
		loss = loss.sum((1,2)) / mask.sum((1,2)).clamp(min=1)
		
		if balancing == 1:
			loss = loss.mean()
		elif balancing == 2:
			# balance anchors by number of positives too
			anchor_patientcount = triplet_dict['id2patient'][-1].item()+1
			pos_counts = pos_mask.sum(-1)
			loss = loss / pos_counts
			loss = loss.sum() / anchor_patientcount
		else:
			raise NotImplementedError(balancing)
	return loss

def triplet_loss_inner_distance(
		features,	# A x D
		config,
		model_extras=None,
		**kwargs
	):
	distances = torch.cdist(features, features)
	if model_extras is None:
		temperature = config['temperature']
		distances = distances * temperature
	else:
		A = len(features)
		bn = model_extras['classifier']
		temperature = bn.weight.detach()
		distances = bn(distances.reshape(-1,1)).reshape(A, A)
	
	return distances, temperature

def triplet_loss_inner_cosine(
		features,	# A x D
		config,
		model_extras=None,
		**kwargs
	):
	scores = F.cosine_similarity(features.unsqueeze(0), features.unsqueeze(1), dim=-1)
	if model_extras is None:
		temperature = config['temperature']
		scores = scores * temperature
	else:
		A = len(features)
		thermostat = model_extras['classifier']
		temperature = thermostat.weight.detach()
		scores = thermostat(scores.reshape(-1,1)).reshape(A,A)
	
	return -scores, temperature
	
def center_loss(
		features,	# A x D
		triplet_dict
	):
	# K x I
	patient_idx = triplet_dict['patient_groups'].to(features.device)
	patient_mask = patient_idx.bool()
	# K x I x D
	patient_embeds = F.embedding(patient_idx, F.pad(features, (0,0,1,0)))
	# K x 1
	patient_counts = patient_idx.bool().sum(-1, keepdim=True)
	# K x 1 x D
	centroids = (patient_embeds.sum(1) / patient_counts).unsqueeze(1)
	# K x I
	squared_distances = torch.pow(torch.cdist(patient_embeds, centroids),2) + 1e-4
	# 1
	variance = (squared_distances.sum(1) / patient_counts).mean()
	
	return variance
