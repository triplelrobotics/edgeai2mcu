# Lint as: python3
# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
r"""An example of semantic segmentation.

The following command runs this script and saves a new image showing the
segmented pixels at the location specified by `output`:

```
bash examples/install_requirements.sh semantic_segmentation.py

python3 examples/semantic_segmentation.py \
	--model test_data/deeplabv3_mnv2_pascal_quant_edgetpu.tflite \
	--input test_data/bird.bmp \
	--keep_aspect_ratio \
	--output ${HOME}/segmentation_result.jpg
```
"""

import argparse

import numpy as np
from PIL import Image

from pycoral.adapters import common
from pycoral.adapters import segment
from pycoral.utils.edgetpu import make_interpreter

import os
import math
import scipy.ndimage  # for zooming the segmentation result to original image size
import urllib.request
from pycoral.utils.edgetpu import load_edgetpu_delegate
import tflite_runtime.interpreter as tflite

import subprocess

####################################################################
print("--- Running Coral TPU Bash script ---")
script_path = './coral_smart_reset.sh'
try:
	# 使用 subprocess.run 执行 Bash 脚本
	# capture_output=True 可以捕获标准输出和标准错误
	# text=True 表示以文本模式处理输出
	# check=True 表示如果命令返回非零退出码，则抛出 CalledProcessError
	result = subprocess.run(
		['/bin/bash', script_path],
		text=True,
		check=True
	)
	print(result.stdout)
	if result.stderr:
		print(f"Bash script stderr:\n{result.stderr}")
	print("--- Coral TPU preparation successful ---")
except subprocess.CalledProcessError as e:
	print(f"--- Coral TPU preparation FAILED ---")
	print(f"Error: Bash script exited with status code {e.returncode}")
	print(f"Stdout:\n{e.stdout}")
	print(f"Stderr:\n{e.stderr}")
	raise RuntimeError("Coral TPU could not be prepared. Exiting.") from e
except Exception as e:
	print(f"An unexpected error occurred during Coral TPU preparation: {e}")
	raise
####################################################################



### 
# Q: why segmentation task cares about aspect ratio (keep_aspect_ratio param) while classification does not?
# A: segmentation task is about pixel-wise labelling, so it needs to have precise pixel-wise boundaries.
###

# SOLVED: had this error RuntimeError: Encountered unresolved custom op: PosenetDecoderOp.Node number 1 (PosenetDecoderOp) failed to prepare.
# see this info: https://github.com/google-coral/edgetpu/issues/123
# also see here for solution: https://github.com/google-coral/project-bodypix/tree/master/posenet_lib/aarch64
# solution: add a posenet delegate to the interpreter

# TODO: the segmentation result is not properly resized to the original image size,  need to fix it.
# maybe because of stride? 
# check how they did: https://github.com/google-coral/project-bodypix/blob/master/bodypix.py

# BodyPix身体部位定义
# mainly used for bodypix models
BODYPIX_PARTS = {
  0: "left face",
  1: "right face", 
  2: "left upper arm front",
  3: "left upper arm back",
  4: "right upper arm front",
  5: "right upper arm back",
  6: "left lower arm front",
  7: "left lower arm back",
  8: "right lower arm front", 
  9: "right lower arm back",
  10: "left hand",
  11: "right hand",
  12: "torso front",
  13: "torso back",
  14: "left upper leg front",
  15: "left upper leg back",
  16: "right upper leg front",
  17: "right upper leg back", 
  18: "left lower leg front",
  19: "left lower leg back",
  20: "right lower leg front",
  21: "right lower leg back",
  22: "left feet",
  23: "right feet",
}

DEVICE = 'usb:0'
MODEL_LIST = {
		'unet-mobilenet-v2-s': {'fn': 'keras_post_training_unet_mv2_128_quant_edgetpu.tflite', 'im_sz': 128, 'labl': 'pet'},
		'unet-mobilenet-v2-l': {'fn': 'keras_post_training_unet_mv2_256_quant_edgetpu.tflite', 'im_sz': 256, 'labl': 'pet'},
		'deeplabv3-mobilenet-v2-s': {'fn': 'deeplabv3_mnv2_dm05_pascal_quant_edgetpu.tflite', 'im_sz': 513, 'labl': 'pascal'},
		'deeplabv3-mobilenet-v2-l': {'fn': 'deeplabv3_mnv2_pascal_quant_edgetpu.tflite', 'im_sz': 513, 'labl': 'pascal'},
		'edgemobilenet-deeplab': {'fn': 'deeplab_mobilenet_edgetpu_slim_cityscapes_quant_edgetpu.tflite', 'im_sz': 513, 'labl': 'cityscapes'},
		'mobilenet-v1-bodypix-324-324': {
			'link': 'https://raw.githubusercontent.com/google-coral/coralmicro/main/models',
			'fn': 'bodypix_mobilenet_v1_075_324_324_16_quant_decoder_edgetpu.tflite', 'im_sz': [324, 324], 'labl': 'bodypix'},
		'mobilenet-v1-bodypix-480-352': {
			'link': 'https://raw.githubusercontent.com/google-coral/project-bodypix/master/models', 
			'fn': 'bodypix_mobilenet_v1_075_480_352_16_quant_decoder_edgetpu.tflite', 'im_sz': [352, 480], 'labl': 'bodypix'},
		'mobilenet-v1-bodypix-512-512': {
			'link': 'https://raw.githubusercontent.com/google-coral/project-bodypix/master/models',
			'fn': 'bodypix_mobilenet_v1_075_512_512_16_quant_decoder_edgetpu.tflite', 'im_sz': [512, 512], 'labl': 'bodypix'},
		'mobilenet-v1-bodypix-640-480': {
			'link': 'https://raw.githubusercontent.com/google-coral/project-bodypix/master/models',
			'fn': 'bodypix_mobilenet_v1_075_640_480_16_quant_decoder_edgetpu.tflite', 'im_sz': [480, 640], 'labl': 'bodypix'},
		'mobilenet-v1-bodypix-768-576': {
			'link': 'https://raw.githubusercontent.com/google-coral/project-bodypix/master/models',
			'fn': 'bodypix_mobilenet_v1_075_768_576_16_quant_decoder_edgetpu.tflite', 'im_sz': [576, 768], 'labl': 'bodypix'},
		'mobilenet-v1-bodypix-1024-768': {
			'link': 'https://raw.githubusercontent.com/google-coral/project-bodypix/master/models',
			'fn': 'bodypix_mobilenet_v1_075_1024_768_16_quant_decoder_edgetpu.tflite', 'im_sz': [768, 1024], 'labl': 'bodypix'},
		'mobilenet-v1-bodypix-1280-720': {
			'link': 'https://raw.githubusercontent.com/google-coral/project-bodypix/master/models',
			'fn': 'bodypix_mobilenet_v1_075_1280_720_16_quant_decoder_edgetpu.tflite', 'im_sz': [720, 1280], 'labl': 'bodypix'},
		'resnet-50-bodypix-416-288': {
			'link': 'https://raw.githubusercontent.com/google-coral/project-bodypix/master/models',
			'fn': 'bodypix_resnet_50_416_288_16_quant_decoder_edgetpu.tflite', 'im_sz': [288, 416], 'labl': 'bodypix'},
		'resnet-50-bodypix-640-480': {
			'link': 'https://raw.githubusercontent.com/google-coral/project-bodypix/master/models',
			'fn': 'bodypix_resnet_50_640_480_16_quant_decoder_edgetpu.tflite', 'im_sz': [480, 640], 'labl': 'bodypix'},
		'resnet-50-bodypix-768-496': {
			'link': 'https://raw.githubusercontent.com/google-coral/project-bodypix/master/models',
			'fn': 'bodypix_resnet_50_768_496_32_quant_decoder_edgetpu.tflite', 'im_sz': [496, 768], 'labl': 'bodypix'},
		'resnet-50-bodypix-864-624': {
			'link': 'https://raw.githubusercontent.com/google-coral/project-bodypix/master/models',
			'fn': 'bodypix_resnet_50_864_624_32_quant_decoder_edgetpu.tflite', 'im_sz': [624, 864], 'labl': 'bodypix'},
		'resnet-50-bodypix-928-672': {
			'link': 'https://raw.githubusercontent.com/google-coral/project-bodypix/master/models',
			'fn': 'bodypix_resnet_50_928_672_16_quant_decoder_edgetpu.tflite', 'im_sz': [672, 928], 'labl': 'bodypix'}, 
		'resnet-50-bodypix-960-736': {
			'link': 'https://raw.githubusercontent.com/google-coral/project-bodypix/master/models',
			'fn': 'bodypix_resnet_50_960_736_32_quant_decoder_edgetpu.tflite', 'im_sz': [736, 960], 'labl': 'bodypix'}
}

PRELOADED_DELEGATE = load_edgetpu_delegate(options={'device': DEVICE})
POSENET_DELEGATE = tflite.load_delegate('test_data_segmentation/posenet_lib/aarch64/posenet_decoder.so')
print("delegates successfully loaded")

# label index mapped to varying colors (total 256 colors available, maximum 256 labels applicable), 
# e.g. 0 -> RGB(0, 0, 0), 1 -> RGB(128, 0, 0), etc.
### 
# input: None
# Output: a 256x3 numpy array with RGB values for each label index
###
def create_pascal_label_colormap():
	"""Creates a label colormap used in PASCAL VOC segmentation benchmark.

	Returns:
		A Colormap for visualizing segmentation results.
	"""
	colormap = np.zeros((256, 3), dtype=int)
	indices = np.arange(256, dtype=int)

	for shift in reversed(range(8)):
		for channel in range(3):
			colormap[:, channel] |= ((indices >> channel) & 1) << shift
		indices >>= 3

	return colormap

###
# input - label: a 2D numpy array with integer type, storing the segmentation label
# Output - result: a 2D numpy array with floating type. The element of the array
###
def label_to_color_image(label):
	"""Adds color defined by the dataset colormap to the label.

	Args:
		label: A 2D array with integer type, storing the segmentation label.

	Returns:
		result: A 2D array with floating type. The element of the array
			is the color indexed by the corresponding element in the input label
			to the PASCAL color map.

	Raises:
		ValueError: If label is not of rank 2 or its value is larger than color
			map maximum entry.
	"""
	if label.ndim != 2:
		raise ValueError('Expect 2-D input label')

	colormap = create_pascal_label_colormap()

	if np.max(label) >= len(colormap):
		raise ValueError('label value too large.')

	return colormap[label]

# specifially for bodypix models. model output is logits, turn logits to probabilities.
def softmax(y, axis):
	"""Softmax function for probability normalization."""
	y = y - np.expand_dims(np.max(y, axis=axis), axis)
	y = np.exp(y)
	return y / np.expand_dims(np.sum(y, axis=axis), axis)

# specifially for bodypix models. 
def calc_stride(h, w, L):
	"""Calculate stride size for BodyPix models."""
	return int((2*h*w)/(math.sqrt(h**2 + 4*h*L*w - 2*h*w + w**2) - h - w))

def process_bodypix_output(interpreter, model_input_height, model_input_width, image):
	"""Process BodyPix model output to get body parts segmentation."""
	"""
	Args:
		interpreter: The TFLite interpreter for the BodyPix model.
		model_input_height: The height of bodypix model's specified input.
		model_input_width: The width of bodypix model's specified input.
	"""
	# 查看所有输出详情
	output_details = interpreter.get_output_details()
	for i, detail in enumerate(output_details):
		print(f"Output {i}: {detail}")

	# # Get both heatmap and parts details from interpreter
	body_details = interpreter.get_output_details()[5]
	body_raw = interpreter.get_tensor(body_details['index'])
	body_shape = body_details['shape']

	# # Get quantization parameters
	body_zero_point = body_details['quantization_parameters']['zero_points'][0]
	body_scale = body_details['quantization_parameters']['scales'][0]

	body_logits = (body_raw.astype(np.float32) - body_zero_point) * body_scale
	assert np.product(body_shape) == body_logits.size
	body_logits = np.reshape(body_logits, body_shape)
	print(body_logits.shape)
	
	rescale_factor = (1, model_input_height / body_shape[1], model_input_width / body_shape[2], 1)
	resized_body_logits = scipy.ndimage.zoom(body_logits, rescale_factor, order=1)
	print(f"resized_body_logits shape: {resized_body_logits.shape}")
	
	# 关键：处理单通道图像
	resize_img = resized_body_logits[0]
	if resize_img.shape[-1] == 1:
		resize_img = np.reshape(resize_img, resize_img.shape[:2])
		
	resize_img = resize_img.astype(np.uint8)
	pil_img = Image.fromarray(resize_img, mode='L')
	pil_img.save('high_res_from_npy.png')
	

	# output_image = image + rgb_heatmap
	# int_img = np.uint8(np.clip(output_image,0,255))
	# return int_img
	return 1

def prepare_data(models_to_test, data_folder):
	# Check if files for test are in the folder
	for model_name in models_to_test:
		model_filename = MODEL_LIST[model_name]['fn']
		label_name = MODEL_LIST[model_name]['labl']
		if label_name == 'pascal':
			img_filename = 'bird_segmentation.bmp'
		elif label_name == 'cityscapes':
			img_filename = 'cityscapes_segmentation.bmp'
		elif label_name == 'bodypix':
			img_filename = 'test_couple.jpg'
		elif label_name == 'pet':
			img_filename = 'pets.jpg'
		
		for fi, fn in enumerate([model_filename, img_filename]):
			file_path = os.path.join(data_folder, fn)
			if not os.path.exists(file_path) and not os.path.isfile(file_path):
				print(f"file {file_path} not found in data folder {data_folder}, prepare to download it ...")
				# download the file
				if fi == 0:
					link = MODEL_LIST[model_name].get('link', 'https://raw.githubusercontent.com/google-coral/test_data/master')
					url = f"{link}/{model_filename}"
					print(f"preparing to download model ...")
				elif fi == 1:
					url = f"https://raw.githubusercontent.com/google-coral/test_data/master/{img_filename}"
					print(f"preparing to download test img ...")
				else:
					raise ValueError(f"invalid file name {fn} to download")
				# download the file
				print(f"downloading from {url}...")
				urllib.request.urlretrieve(url, file_path)
				print(f"downloaded")
			else:
				print(f"file {file_path} found in data folder {data_folder}, skipping download ...")
	
	print(f"all files are ready in {data_folder} folder, ready to run inference ...")

def run_inference(models_to_test, data_folder, keep_aspect_ratio):
	for model_name in models_to_test:
		model_filename = MODEL_LIST[model_name]['fn']
		label_name = MODEL_LIST[model_name]['labl']
		if label_name == 'pascal':
			img_filename = 'bird_segmentation.bmp'
		elif label_name == 'cityscapes':
			img_filename = 'cityscapes_segmentation.bmp'
		elif label_name == 'bodypix':
			img_filename = 'test_couple.jpg'
			# img_filename = 'two_persons.png' # use a couple image for bodypix models
		elif label_name == 'pet':
			img_filename = 'pets.jpg'

		model_filepath = os.path.join(data_folder, model_filename)
		img_filepath = os.path.join(data_folder, img_filename)

		# create interpreter
		if label_name == 'bodypix': # use both delegates for BodyPix models
			interpreter = tflite.Interpreter(
				model_path=model_filepath, 
				experimental_delegates=[PRELOADED_DELEGATE, POSENET_DELEGATE]
			)
		else:
			interpreter = make_interpreter(model_filepath, device=DEVICE, delegate=PRELOADED_DELEGATE)
		
		interpreter.allocate_tensors()
		width, height = common.input_size(interpreter)

		img = Image.open(img_filepath)
		if keep_aspect_ratio:
			resized_img, _ = common.set_resized_input(interpreter, img.size, lambda size: img.resize(size, Image.LANCZOS))
		else:
			resized_img = img.resize((width, height), Image.LANCZOS)
			common.set_input(interpreter, resized_img)

		interpreter.invoke()

		# Process output based on model type
		if label_name == 'bodypix': # extra process for bodypix models
			result = process_bodypix_output(interpreter, height, width, np.array(resized_img))
		else:
			result = segment.get_output(interpreter)
			if len(result.shape) == 3:
				result = np.argmax(result, axis=-1)

		# If keep_aspect_ratio, we need to remove the padding area.
		new_width, new_height = resized_img.size
		result = result[:new_height, :new_width]

		if label_name == 'bodypix':
			mask_img = Image.fromarray(result.astype(np.uint8))
		else:
			mask_img = Image.fromarray(label_to_color_image(result).astype(np.uint8))
		
		# Concat resized input image and processed segmentation results.
		output_img = Image.new('RGB', (2 * new_width, new_height))
		output_img.paste(resized_img, (0, 0))
		output_img.paste(mask_img, (width, 0))
		
		save_img_name = img_filename.split(".")[0]
		output_path = os.path.join(data_folder, f"{model_name}_{save_img_name}_segmentation_result_3.jpg")
		output_img.save(output_path)
		
		print(f'Done for {model_name}. Results saved at {output_path}')

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--model_name', required=True, help='a model nickname, e.g. "deeplabv3-mobilenet-v2-s"')
	parser.add_argument('--data_folder', type=str, default='test_data_segmentation', help='data folder for downloading models and tesst images')
	parser.add_argument('--keep_aspect_ratio', action='store_true', default=False, help=(
					'keep the image aspect ratio when down-sampling the image by adding '
					'black pixel padding (zeros) on bottom or right. '
					'By default the image is resized and reshaped without cropping. This '
					'option should be the same as what is applied on input images during '
					'model training. Otherwise the accuracy may be affected and the '
					'bounding box of detection result may be stretched.'))
	args = parser.parse_args()

	# assert model name param can be found from list
	models_to_test = []
	if args.model_name == 'all':
		models_to_test = list(MODEL_LIST.keys())
		print(f"testing {len(models_to_test)} models")
	else:
		if args.model_name not in MODEL_LIST:
			raise ValueError(f"invalid model name, please choose a name from this list {MODEL_LIST.keys}")
		models_to_test = [args.model_name]

	# if not specified data folder, create it
	if not os.path.exists(args.data_folder):
		os.makedirs(args.data_folder)
		print(f"creating data folder: {args.data_folder}")

	# prepare data
	prepare_data(models_to_test, args.data_folder)

	# run inference 
	run_inference(models_to_test, args.data_folder, args.keep_aspect_ratio)

if __name__ == '__main__':
	main()