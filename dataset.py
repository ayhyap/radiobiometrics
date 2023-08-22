import os
import torch
import torchvision
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image, ImageOps
from torchvision import transforms

# creates batches of PATIENTS of images
# all studies are unrolled to yield a sequence of images
def patient_collate_flatten(inputs):
	''' 
	INPUTS (list of dictionaries)
		each dictionary has variable number of images
		images have fixed dimensions
			patients x var(images) x c x h x w
	
	OUTPUTS (at least the important ones)
	images			total(images) x c x h x w
	id2patient		total(images)
	''' 
	outputs = {}
	
	# this list comprehension flattens the variable-length multidimensional lists
	# total(images) x c x h x w
	outputs['images'] = torch.stack([img for patient in inputs for img in patient['images']])
	if inputs[0]['orig_images'] is not None:
		outputs['orig_images'] = torch.stack([img for patient in inputs for img in patient['orig_images']])
	
	# make mappings from flattened index to patient ID (reset to 0,1,2,...)
	patient_image_counts = np.array([len(patient['images']) for patient in inputs])
	id2patient = np.zeros(len(outputs['images']), dtype=int)
	'''
	counts
	1 3 2
	cumsum [:-1]
	1 4
	indexed on zeros
	0 1 0 0 1 0
	cumsum
	0 1 1 1 2 2
	'''
	_cumsum = patient_image_counts.cumsum()
	id2patient[_cumsum[:-1]] = 1
	id2patient = id2patient.cumsum()
	
	# 0 to N-1
	outputs['id2patient'] = torch.from_numpy(id2patient)
	
	# files
	# patients x var(images) -> total(images)
	files = np.array([patient['files'] for patient in inputs], dtype=object)
	outputs['files'] = np.concatenate(files).tolist()
	outputs['idx'] = torch.tensor([idx for patient in inputs for idx in patient['idx']])
	return outputs

# helper class for padding to square using torchvision transforms
class SquarePad():
	def __call__(self, image):
		w, h = image.size
		max_wh = np.max([w, h])
		hp = int((max_wh - w) / 2)
		vp = int((max_wh - h) / 2)
		padding = (hp, vp, hp, vp)
		return transforms.functional.pad(image, padding, 0, 'constant')

class BBoxCrop():
	def __call__(self, image):
		return image.crop(image.getbbox())

class HistEQ():
	def __call__(self, image):
		return ImageOps.equalize(image)

'''
PARAMETERS
meta_df: pd.DataFrame	metadata dataframe (has 1 entry per image)
img_dir: str			directory that contains images
img_size: int			size to resize images to (length of square)
augment: int			image augmentation intensity (0,1,2)
augment_pairs: bool		whether to augment images twice for image-contrastive learning
augment_singleton: bool	whether to augment images twice for patients with only 1 image
img_limit: int			number of images to limit per patient to avoid overloading GPU memory
return_orig_img: bool	whether to return original images for manual inspection
'''
class ContrastivePatientDS(Dataset):
	def __init__(	self, 
					meta_df,
					img_dir, 
					img_size, 
					augment,
					augment_pairs,
					augment_singleton=False,
					img_limit=10,
					return_orig_img=False
				):
		self.meta_df = meta_df
		self.img_dir = img_dir
		self.image_limit = img_limit
		self.augment_pairs = augment_pairs
		self.augment_singleton = augment_singleton
		self.return_orig_img = return_orig_img
		
		self.patients = self.meta_df.subject_id.unique()
		
		## construct preprocessor
		# note: dataloading can be sped up by preprocessing images with bounding box crop, histogram equalization, and resizing
		# if preprocessed, comment the bboxcrop() and histeq() lines (also the ones for self.basic_preprocess!)
		preprocess = []
		preprocess.append(BBoxCrop())
		preprocess.append(HistEQ())
		preprocess.append(SquarePad())
		preprocess.append(transforms.Resize((img_size,img_size)))
		
		if augment == 2:
			# heavier augmentation
			preprocess.append(transforms.ColorJitter(brightness=0.2, contrast=0.2))
			preprocess.append(transforms.RandomAffine(	20, 
														translate=(0.4,0.4), 
														scale=(0.8,1.2), 
														shear=10, 
														interpolation=torchvision.transforms.InterpolationMode.BILINEAR, 
														fill=0))
		elif augment == 1:
			# milder augmentation
			preprocess.append(transforms.ColorJitter(brightness=0.1, contrast=0.1))
			preprocess.append(transforms.RandomAffine(	10, 
														translate=(0.2,0.2), 
														scale=(0.9,1.1), 
														shear=5, 
														interpolation=torchvision.transforms.InterpolationMode.BILINEAR, 
														fill=0))
		elif augment == 0:
			pass
		else:
			raise ValueError(augment)
		
		preprocess += [transforms.ToTensor(),
					   transforms.Normalize( (0.502,) , (0.289,) )]
		
		self.preprocess = transforms.Compose(preprocess)
		self.basic_preprocess = transforms.Compose([
									BBoxCrop(),
									HistEQ(),
									SquarePad(), 
									transforms.Resize((img_size,img_size)), 
									transforms.ToTensor()])
	
	def __len__(self):
		return len(self.patients)
	
	def __getitem__(self,dataset_index):
		'''
		EXPECTED OUTPUTS
		images			var(images) x c x h x w
		files			total_images
		augment_pairs	1
		'''
		# get relevant rows
		patient = self.patients[dataset_index]
		patient_rows = self.meta_df[self.meta_df.subject_id == patient]
		
		# if too many images, randomly sample N images
		if len(patient_rows) > self.image_limit:
			patient_rows = patient_rows.sample(self.image_limit)
		
		orig_images = [] if self.return_orig_img else None
		images = []
		files = []
		
		for i, row in patient_rows.iterrows():
			file = '{}/{}'.format(self.img_dir,row.path)
			try:
				img = Image.open(file).convert('RGB')
			except FileNotFoundError:
				continue
			
			images.append(self.preprocess(img))
			files.append(file)
			if self.return_orig_img:
				orig_images.append(self.basic_preprocess(img))
			
			if self.augment_pairs:
				images.append(self.preprocess(img))
				files.append(file)
				if self.return_orig_img:
					orig_images.append(self.basic_preprocess(img))
		
		if len(images) == 1 and self.augment_singleton:
			assert img is not None
			images.append(self.preprocess(img))
			files.append(file)
			if self.return_orig_img:
				orig_images.append(self.basic_preprocess(img))
		
		
		outputs = {}
		
		outputs['orig_images'] = orig_images
		outputs['images'] = images
		outputs['files'] = files
		outputs['augment_pairs'] = self.augment_pairs
		outputs['idx'] = [dataset_index for _ in files]
		return outputs