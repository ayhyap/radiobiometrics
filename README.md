# ((placeholder))

## Overview
Code repository for paper currently under review.
For confidentiality reasons, minimal details will be provided until later notice.

## Usage
### Installation
- Clone/download the repository, then cd into it.
- Use anaconda to set up an environment.
`conda env create -f environment`

### Running
- Activate the conda environment
`conda activate radiobiom`

#### Data Preprocessing
Because images from MIMIC-CXR-JPG are full-sized, it is **highly recommended** to preprocess the images to reduce train-time preprocessing and disc I/O. 
We apply:
1. Bounding Box Crop `PIL.Image.crop(PIL.Image.getbbox())`
2. Histogram Equalization `cv2.equalizehist()` or `PIL.ImageOps.equalize()`
3. Resizing to 256 x ? or ? x 256 while maintaining aspect ratio
4. Zero-padding to 256 x 256

For ease of use, the provided code applies the preprocessing steps at train-time.
If you have preprocessed the images as described above, you may comment out the relevant lines in *dataset.py* with `BBoxCrop()` and `HistEQ()`.

#### Training a model
The provided code was done for MIMIC-CXR-JPG and will require editing for other datasets.

- Set up your data directories in *globals.json*, specifically `image_dir`, `image_df`, and `split_df`
> {
    "image_dir" : "path/to/image/dir",
    "image_df" : "path/to/mimic-cxr-2.0.0-metadata.csv",
    "split_df" : "path/to/mimic-cxr-2.0.0-split.csv",
    ...
}
- If necessary, edit *main.py* to work on your data (around line #84)
- Create a new experiment `.json` configuration file in the `runs` folder (or just use baseline_short.json)

All settings (model hyperparameters, length of training, etc.) are defaulted to those in `runs/baseline.json`. You only need to add whatever settings you want to override.

For now there will be no explanations for each setting, so inspect the code for valid settings.

- run or nohup, etc. `main.py [configuration json] [cpu/0/1/2/...]`

1st parameter is the path to the configuration file in the previous step.
2nd parameter is the device to use. To use GPU, use 0. To use a specific GPU on a device with multiple GPUs, use the desired index. Multiple GPU usage is not supported.
Using baseline settings requires 16GB of GPU memory.

#### Testing

The code provided is currently setup to test on the same validation/test data used during training.
You will need to similarly set up directories in *globals.json* and edit *test.py* to work on your data.

- run or nohup, etc. `test.py [configuration json] [cpu/0/1/2/...]`

The parameters are the same as the ones used for training in *main.py*.
The script will output pickle files to the same directory of the saved model.
This script expects a trained model which follows the configuration provided.


(2023/09/05) UPDATE:
You can now run test_folder.py to test a trained model on a folder of images (jpg, png, etc.; dicom not supported), specifically the 'test_images'.

Like above, run `test_folder.py [configuration json] [cpu/0/1/2/...]`

Under 'test_images', create subfolders for each subject and place their images in.

Subjects with only 1 image can be placed directly into the 'test_images' folder as loose images.

50 images from Radiopaedia are provided in this repository. Images are used courtesy of the respective parties under the CC license.

Access the original images at: radiopaedia.org/cases/[case number]

The case number is the file name for loose images, and the folder name for others.

## Others
Our experiments on MIMIC-CXR used a filtered and preprocessed subset of data which we will not disclose in public for confidentiality reasons.
Contact us directly for more details.
