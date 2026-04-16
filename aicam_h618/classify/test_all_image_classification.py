import argparse
import time
import urllib.request
import os

import numpy as np
from PIL import Image
from pycoral.adapters import classify
from pycoral.adapters import common
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter
from pycoral.utils.edgetpu import load_edgetpu_delegate

# TODO: configure for popular-100k model and its input.

DEVICE = 'usb:0'
MODEL_LIST = {
	'efficientnet-edgetpu-l': {'fn': 'efficientnet-edgetpu-L_quant_edgetpu.tflite', 'im_sz': 300, 'labl': 'imagenet'},
	'efficientnet-edgetpu-m': {'fn': 'efficientnet-edgetpu-M_quant_edgetpu.tflite', 'im_sz': 240, 'labl': 'imagenet'},
	'efficientnet-edgetpu-s': {'fn': 'efficientnet-edgetpu-S_quant_edgetpu.tflite', 'im_sz': 224, 'labl': 'imagenet'},
	'inception-v1': {'fn': 'inception_v1_224_quant_edgetpu.tflite', 'im_sz': 224, 'labl': 'imagenet'},
	'inception-v2': {'fn': 'inception_v2_224_quant_edgetpu.tflite', 'im_sz': 224, 'labl': 'imagenet'},
	'inception-v3': {'fn': 'inception_v3_299_quant_edgetpu.tflite', 'im_sz': 299, 'labl': 'imagenet'},
	'inception-v4': {'fn': 'inception_v4_299_quant_edgetpu.tflite', 'im_sz': 299, 'labl': 'imagenet'}, 
	'mobilenet-v1-ss': {'fn': 'mobilenet_v1_0.25_128_quant_edgetpu.tflite', 'im_sz': 128, 'labl': 'imagenet'},
	'mobilenet-v1-s': {'fn': 'mobilenet_v1_0.5_160_quant_edgetpu.tflite', 'im_sz': 160, 'labl': 'imagenet'},
	'mobilenet-v1-m': {'fn': 'mobilenet_v1_0.75_192_quant_edgetpu.tflite', 'im_sz': 192, 'labl': 'imagenet'},
	'mobilenet-v1-l': {'fn': 'mobilenet_v1_1.0_224_quant_edgetpu.tflite', 'im_sz': 224, 'labl': 'imagenet'},
	'mobilenet-v1-l-tf2': {'fn': 'tf2_mobilenet_v1_1.0_224_ptq_edgetpu.tflite', 'im_sz': 224, 'labl': 'imagenet'},
	'mobilenet-v2-inet-birds': {'fn': 'mobilenet_v2_1.0_224_inat_bird_quant_edgetpu.tflite', 'im_sz': 224, 'labl': 'inat_bird'},
	'mobilenet-v2-inet-insects': {'fn': 'mobilenet_v2_1.0_224_inat_insect_quant_edgetpu.tflite', 'im_sz': 224, 'labl': 'inat_insect'},
	'mobilenet-v2-inet-plants': {'fn': 'mobilenet_v2_1.0_224_inat_plant_quant_edgetpu.tflite', 'im_sz': 224, 'labl': 'inat_plant'},
	'mobilenet-v2': {'fn': 'mobilenet_v2_1.0_224_quant_edgetpu.tflite', 'im_sz': 224, 'labl': 'imagenet'},
	'mobilenet-v2-tf2': {'fn': 'tf2_mobilenet_v2_1.0_224_ptq_edgetpu.tflite', 'im_sz': 224, 'labl': 'imagenet'},
	'mobilenet-v3-tf2': {'fn': 'tf2_mobilenet_v3_edgetpu_1.0_224_ptq_edgetpu.tflite', 'im_sz': 224,  'labl': 'imagenet'},
	'resnet-50': {'fn': 'tfhub_tf2_resnet_50_imagenet_ptq_edgetpu.tflite', 'im_sz': 224, 'labl': 'imagenet'},
	# 'popular-100k': {'fn': 'tfhub_tf1_popular_us_products_ptq_fc_split_edgetpu.tflite', 'im_sz': 224, 'labl': 'popular_100k'},
	}

PRELOADED_DELEGATE = load_edgetpu_delegate(options={'device': DEVICE})

def prepare_data(models_to_test, data_folder):
	# Check if files for test are in the folder
	for model_name in models_to_test:
		model_filename = MODEL_LIST[model_name]['fn']
		label_filename = MODEL_LIST[model_name]['labl'] + '_labels.txt'
		img_filename = 'parrot.jpg' if label_filename == 'imagenet_labels.txt' else 'dragonfly.bmp' if label_filename == 'inat_insect_labels.txt' else 'sunflower.bmp'
		
		for fi, fn in enumerate([model_filename, label_filename, img_filename]):
			file_path = os.path.join(data_folder, fn)
			if not os.path.exists(file_path) and not os.path.isfile(file_path):
				print(f"file {file_path} not found in data folder {data_folder}, prepare to download it ...")
				# download the file
				if fi == 0:
					url = f"https://raw.githubusercontent.com/google-coral/test_data/master/{model_filename}"
					print(f"preparing to download model ...")
				elif fi == 1:
					url = f"https://raw.githubusercontent.com/google-coral/test_data/master/{label_filename}"
					print(f"preparing to download label file ...")
				elif fi == 2:
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


def run_inference(models_to_test, data_folder, top_k, threshold, count, input_mean, input_std):
	# Loop through each model and run inference
	for model_name in models_to_test:
		model_filename = MODEL_LIST[model_name]['fn']
		label_filename = MODEL_LIST[model_name]['labl'] + '_labels.txt'
		img_filename = 'parrot.jpg' if label_filename == 'imagenet_labels.txt' else 'dragonfly.bmp' if label_filename == 'inat_insect_labels.txt' else 'sunflower.bmp'
		
		model_filepath = os.path.join(data_folder, model_filename)
		label_filepath = os.path.join(data_folder, label_filename)
		img_filepath = os.path.join(data_folder, img_filename)

		# create label object
		labels = read_label_file(label_filepath)
		
		# create interpreter
		interpreter = make_interpreter(*model_filepath.split('@'), device=DEVICE, delegate=PRELOADED_DELEGATE)
		interpreter.allocate_tensors()

		# Model must be uint8 quantized
		if common.input_details(interpreter, 'dtype') != np.uint8:
			raise ValueError('Only support uint8 input type.')

		size = common.input_size(interpreter)
		image = Image.open(img_filepath).convert('RGB').resize(size, Image.LANCZOS)

		# Image data must go through two transforms before running inference:
		# 1. normalization: f = (input - mean) / std
		# 2. quantization: q = f / scale + zero_point
		# The following code combines the two steps as such:
		# q = (input - mean) / (std * scale) + zero_point
		# However, if std * scale equals 1, and mean - zero_point equals 0, the input
		# does not need any preprocessing (but in practice, even if the results are
		# very close to 1 and 0, it is probably okay to skip preprocessing for better
		# efficiency; we use 1e-5 below instead of absolute zero).
		params = common.input_details(interpreter, 'quantization_parameters')
		scale = params['scales']
		zero_point = params['zero_points']
		mean = input_mean
		std = input_std

		###
		# Q: why need to check this quantization parameters for input?
		# A: because trained models is quantized (weights float -> int), and input data must be quantized too (1. mean/std norm 2. float->int by scale and zero_point). 
		if abs(scale * std - 1) < 1e-5 and abs(mean - zero_point) < 1e-5:
			# Input data does not require preprocessing.
			common.set_input(interpreter, image)
		else:
			# Input data requires preprocessing
			normalized_input = (np.asarray(image) - mean) / (std * scale) + zero_point
			np.clip(normalized_input, 0, 255, out=normalized_input)
			common.set_input(interpreter, normalized_input.astype(np.uint8))

		# Run inference
		print('----INFERENCE TIME----')
		print(f'Model name: {model_name}')
		print('Note: The first inference on Edge TPU is slow because it includes',
		'loading the model into Edge TPU memory.')
		for _ in range(count):
			start = time.perf_counter()
			interpreter.invoke()
			inference_time = time.perf_counter() - start
			classes = classify.get_classes(interpreter, top_k, threshold)
			print('%.1fms' % (inference_time * 1000))

		print('-------RESULTS--------')
		for c in classes:
			print('%s: %.5f' % (labels.get(c.id, c.id), c.score))


def main():
	parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('-m', '--model_name', required=True, help='inference model name.') # model name, matching MODEL_LIST dict above; if 'all', test all models at once.
	parser.add_argument('-d', '--data_folder', type=str, default='test_data_classification', help='data folder name')
	parser.add_argument('-k', '--top_k', type=int, default=1, help='Max number of classification results')
	parser.add_argument('-t', '--threshold', type=float, default=0.0, help='Classification score threshold')
	parser.add_argument('-c', '--count', type=int, default=5, help='Number of times to run inference')
	parser.add_argument('-a', '--input_mean', type=float, default=128.0, help='Mean value for input normalization')
	parser.add_argument('-s', '--input_std', type=float, default=128.0, help='STD value for input normalization')
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
	run_inference(models_to_test, args.data_folder, args.top_k, args.threshold, args.count, args.input_mean, args.input_std)

if __name__ == '__main__':
	main()