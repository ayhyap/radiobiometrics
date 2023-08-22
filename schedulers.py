import torch

'''
manually done schedulers
saves model checkpoints to config['_savedir']
	
implements these functions:

__init__(
	config: dict, 
	globals: dict
)
check(
	model: torch.nn.Module, 
	score: float, 
	checkpoint: int, 
	iter: int
)
returns True if training should continue, else False
'''

# keep training until out of patience (no improvement), then decay LR and reset patience
class PatienceDecay():
	def __init__(self, config, globals):
		self.decays = config['decay_count']
		self.decay_factor = config['decay_factor']
		self.current_factor = 1
		
		self.max_patience = config['patience']
		self.patience = self.max_patience
		self.best_score = 0.0
		
		self.config = config
	
	def decay(self, model):
		checkpoint = torch.load('{}/best.pt'.format(self.config['_savedir']))
		model.load_state_dict(checkpoint['model_state_dict'])
		model.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
		self.current_score = checkpoint['current_score']
		self.current_factor /= self.decay_factor
		for group in model.optimizer.param_groups:
			group['lr'] /= self.decay_factor
		self.patience = self.max_patience
		return model
	
	
	# returns True if training should continue, else False
	def check(self, model, score, checkpoint, iter):
		if score > self.best_score:
			self.best_score = score
			# save best
			savedict = {
				'checkpoint': checkpoint,
				'iteration': iter,
				'model_state_dict': model.state_dict(),
				'scheduler': 'PatienceDecay',
				'decays': self.decays,
				'decay_factor': self.decay_factor,
				'current_factor': self.current_factor,
				'max_patience': self.max_patience,
				'patience': self.patience
			}
			torch.save(savedict, '{}/best.pt'.format(self.config['_savedir']))
			model.best = True
			self.patience = self.max_patience
		else:
			self.patience -= 1
			model.best = False
			if self.patience < 0:
				self.decays -= 1
				if self.decays < 0:
					self.patience = self.decays = -1
				else:
					model = self.decay(model)
		return model, (self.patience >= 0)



# basically stepLR, but extends the end of each LR step until out of patience
class MinLengthPatienceDecay():
	def __init__(self, config, globals):
		self.decay_factor = config['decay_factor']
		self.current_factor = 1
		
		self.max_patience = config['patience']
		self.decay_lengths = config['decay_lengths']
		self.decays = len(self.decay_lengths)-1
		
		self.patience = self.max_patience
		self.current_decay_checkpoints = 0
		
		self.best_score = 0.0
		self.config = config
		
	def decay(self, model):
		checkpoint = torch.load('{}/best.pt'.format(self.config['_savedir']))
		model.load_state_dict(checkpoint['model_state_dict'])
		model.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
		self.current_factor /= self.decay_factor
		for group in model.optimizer.param_groups:
			group['lr'] /= self.decay_factor
		
		self.patience = self.max_patience
		self.current_decay_checkpoints = 0
		return model
	
	# returns True if training should continue, else False
	def check(self, model, score, checkpoint, iter):
		self.current_decay_checkpoints += 1
		
		if score > self.best_score:
			self.best_score = score
			# save best
			savedict = {
				'checkpoint': checkpoint,
				'iteration': iter,
				'model_state_dict': model.state_dict(),
				'scheduler': 'MinLengthPatienceDecay',
				'decay_factor': self.decay_factor,
				'current_factor': self.current_factor,
				'score': score,
			}
			torch.save(savedict, '{}/best.pt'.format(self.config['_savedir']))
			model.best = True
		else:
			model.best = False
		
		if self.current_decay_checkpoints >= self.decay_lengths[len(self.decay_lengths)-self.decays-1]:
			if model.best:
				self.patience = self.max_patience
			else:
				self.patience -= 1
				if self.patience < 0:
					self.decays -= 1
					if self.decays < 0:
						self.patience = self.decays = -1
					else:
						model = self.decay(model)
					
		return model, (self.patience >= 0)


# run for set length at each LR
class StepLR():
	def __init__(self, config, globals):
		self.decay_factor = config['decay_factor']
		self.current_factor = 1
		
		self.decay_lengths = config['decay_lengths']
		self.decays = len(self.decay_lengths)-1
		
		self.current_decay_checkpoints = 0
		
		self.best_score = 0.0
		self.config = config
		
	def decay(self, model):
		checkpoint = torch.load('{}/best.pt'.format(self.config['_savedir']))
		model.load_state_dict(checkpoint['model_state_dict'])
		model.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
		self.current_factor /= self.decay_factor
		for group in model.optimizer.param_groups:
			group['lr'] /= self.decay_factor
		
		self.current_decay_checkpoints = 0
		return model
	
	# returns True if training should continue, else False
	def check(self, model, score, checkpoint, iter):
		self.current_decay_checkpoints += 1
		
		if score > self.best_score:
			self.best_score = score
			# save best
			savedict = {
				'checkpoint': checkpoint,
				'iteration': iter,
				'model_state_dict': model.state_dict(),
				'scheduler': 'StepLR',
				'decay_factor': self.decay_factor,
				'current_factor': self.current_factor,
				'score': score,
			}
			torch.save(savedict, '{}/best.pt'.format(self.config['_savedir']))
			model.best = True
		else:
			model.best = False
		
		if self.current_decay_checkpoints >= self.decay_lengths[len(self.decay_lengths)-self.decays-1]:
			self.decays -= 1
			if self.decays >= 0:
				model = self.decay(model)
			
		return model, (self.decays >= 0)


# run for set length, then continuously linearly decay LR 
class ExploreExploit():
    def __init__(self, config, globals):
        self.explore_count = config['explore_validations']
        self.exploit_count = config['exploit_validations']
        self.current_factor = 1
        
        self.val_counter = 0
        
        self.best_score = 0
        
        self.config = config
    
    def check(self, model, score, checkpoint, iter):
        self.val_counter += 1
        if score > self.best_score:
            self.best_score = score
            # save best
            torch.save({
                'checkpoint': checkpoint,
                'iteration': iter,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': model.optimizer.state_dict(),
                'scheduler': 'ExploreExploit',
                'explore_count': self.explore_count,
                'exploit_count': self.exploit_count,
                'current_factor': self.current_factor,
                'val_counter': self.val_counter
            }, '{}/best.pt'.format(self.config['_savedir']))
            model.best = True
        else:
            model.best = False
        
        if self.val_counter >= self.explore_count:
            if self.val_counter >= (self.explore_count + self.exploit_count):
                # lr should be 0 now, end.
                return model, False
            # decay learning rate
            i = self.val_counter - self.explore_count
            factor = (self.exploit_count-i) / (self.exploit_count+1-i)
            self.current_factor *= factor
            for group in model.optimizer.param_groups:
                group['lr'] *= factor
        
        return model, True