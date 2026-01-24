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
r"""Example using PyCoral to detect objects in a given image.

To run this code, you must attach an Edge TPU attached to the host and
install the Edge TPU runtime (`libedgetpu.so`) and `tflite_runtime`. For
device setup instructions, see coral.ai/docs/setup.

Example usage:
```
bash examples/install_requirements.sh detect_image.py

python3 examples/detect_image.py \
  --model test_data/ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite \
  --labels test_data/coco_labels.txt \
  --input test_data/grace_hopper.bmp \
  --output ${HOME}/grace_hopper_processed.bmp
```
"""

import os
import argparse
import time
import urllib.request # for downloading files

from PIL import Image
from PIL import ImageDraw

from pycoral.adapters import common
from pycoral.adapters import detect
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter
from pycoral.utils.edgetpu import load_edgetpu_delegate  # do this to load lib once for all tests

# do this every time:
# sudo usermod -aG plugdev $USER


DEVICE = 'usb'
# warning: efficientdet-lite3x is not compatible with usb.
MODEL_LIST = {'ssd-mobilenet-v1':{'fn': 'ssd_mobilenet_v1_coco_quant_postprocess_edgetpu.tflite', 'im_sz': 300, 'labl': 'coco'},
                'ssd-mobilenet-v2': {'fn': 'ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite', 'im_sz': 300, 'labl': 'coco'},
                'ssd-mobilenet-v2-tf2': {'fn': 'tf2_ssd_mobilenet_v2_coco17_ptq_edgetpu.tflite', 'im_sz': 300, 'labl': 'coco'},
                'ssd-mobilenet-v2-faces': {'fn': 'ssd_mobilenet_v2_face_quant_postprocess_edgetpu.tflite', 'im_sz': 320, 'labl': 'coco'},
                'ssdlite-mobileDet': {'fn': 'ssdlite_mobiledet_coco_qat_postprocess_edgetpu.tflite', 'im_sz': 320, 'labl': 'coco'},
                'efficientdet-lite0': {'fn': 'efficientdet_lite0_320_ptq_edgetpu.tflite', 'im_sz': 320, 'labl': 'coco'},
                'efficientdet-lite1': {'fn': 'efficientdet_lite1_384_ptq_edgetpu.tflite', 'im_sz': 384, 'labl': 'coco'},
                'efficientdet-lite2': {'fn': 'efficientdet_lite2_448_ptq_edgetpu.tflite', 'im_sz': 448, 'labl': 'coco'},
                'efficientdet-lite3': {'fn': 'efficientdet_lite3_512_ptq_edgetpu.tflite', 'im_sz': 512, 'labl': 'coco'},
                'ssd-fpn-mobilenet-v1-tf2': {'fn': 'tf2_ssd_mobilenet_v1_fpn_640x640_coco17_ptq_edgetpu.tflite', 'im_sz': 640, 'labl': 'coco'}
}
PRELOADED_DELEGATE = load_edgetpu_delegate(options={'device': DEVICE})

def draw_objects(draw, objs, labels):
  """Draws the bounding box and label for each object."""
  for obj in objs:
    bbox = obj.bbox
    draw.rectangle([(bbox.xmin, bbox.ymin), (bbox.xmax, bbox.ymax)],
                   outline='red')
    draw.text((bbox.xmin + 10, bbox.ymin + 10),
              '%s\n%.2f' % (labels.get(obj.id, obj.id), obj.score),
              fill='red')

def prepare_data(models_to_test, data_folder):
  for model_name in models_to_test:
    model_filename = MODEL_LIST[model_name]['fn']
    label_filename = MODEL_LIST[model_name]['labl'] + '_labels.txt'
    img_filename = 'grace_hopper.bmp' if label_filename == 'coco_labels.txt' else None
    if img_filename is None:
      raise ValueError(f"Unknown label file {label_filename} for model {model_name}, please add it to the MODEL_LIST.")

    for fi, fn in enumerate([model_filename, label_filename, img_filename]):
      file_path = os.path.join(data_folder, fn)
      if not os.path.exists(file_path) or not os.path.isfile(file_path):
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
          print(f"preparing to download image ...")
        else:
          raise ValueError(f"invalid file name {fn} to download")
        print(f"downloading from {url}...")
        urllib.request.urlretrieve(url, file_path)
        print(f"download completed: {file_path}")
      else:
        print(f"file {file_path} found in data folder {data_folder}, skipping download ...")
  print(f"all files are ready in {data_folder} folder, ready to run inference ...")
  
def run_inference(models_to_test, data_folder, threshold, count):
  for model_name in models_to_test:
    model_filename = MODEL_LIST[model_name]['fn']
    label_filename = MODEL_LIST[model_name]['labl'] + '_labels.txt'
    img_filename = 'grace_hopper.bmp' if label_filename == 'coco_labels.txt' else None
    if img_filename is None:
      raise ValueError(f"Unknown label file {label_filename} for model {model_name}, please add it to the MODEL_LIST.")
    
    model_filepath = os.path.join(data_folder, model_filename)
    label_filepath = os.path.join(data_folder, label_filename)
    img_filepath = os.path.join(data_folder, img_filename)

    # create label object
    labels = read_label_file(label_filepath)
    
    # create interpreter
    interpreter = make_interpreter(*model_filepath.split('@'), device=DEVICE, delegate=PRELOADED_DELEGATE)
    interpreter.allocate_tensors()

    image = Image.open(img_filepath)
    _, scale = common.set_resized_input(interpreter, image.size, lambda size: image.resize(size, Image.LANCZOS))

    print('----INFERENCE TIME----')
    print(f'Model name: {model_name}')
    print('Note: The first inference is slow because it includes loading the model into Edge TPU memory.')
    for _ in range(count):
      start = time.perf_counter()
      interpreter.invoke()
      inference_time = time.perf_counter() - start
      objs = detect.get_objects(interpreter, threshold, scale)
      print('%.2f ms' % (inference_time * 1000))

    print('-------RESULTS--------')
    if not objs:
      print('No objects detected')

    for obj in objs:
      print(labels.get(obj.id, obj.id))
      print('  id:    ', obj.id)
      print('  score: ', obj.score)
      print('  bbox:  ', obj.bbox)
    
    image = image.convert('RGB')
    draw_objects(ImageDraw.Draw(image), objs, labels)
    image.save(data_folder + f'/{img_filename}_{model_name}_objdetect_result.bmp')
    # image.show()


    



def main():
  parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('-m', '--model_name', required=True, help='inference model name.')
  # parser.add_argument('-i', '--input', required=True, help='File path of image to process')
  parser.add_argument('-i', '--data_folder', default='test_data_detection', help='data folder name')
  # parser.add_argument('-l', '--labels', help='File path of labels file')
  parser.add_argument('-t', '--threshold', type=float, default=0.4, help='Score threshold for detected objects')
  # parser.add_argument('-o', '--output', help='File path for the result image with annotations')
  parser.add_argument('-c', '--count', type=int, default=5, help='Number of times to run inference')
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
  run_inference(models_to_test, args.data_folder, args.threshold, args.count)

  # labels = read_label_file(args.labels) if args.labels else {}
  # interpreter = make_interpreter(args.model)
  # interpreter.allocate_tensors()

  # image = Image.open(args.input)
  # _, scale = common.set_resized_input(
  #     interpreter, image.size, lambda size: image.resize(size, Image.LANCZOS))

  # print('----INFERENCE TIME----')
  # print('Note: The first inference is slow because it includes',
  #       'loading the model into Edge TPU memory.')
  # for _ in range(args.count):
  #   start = time.perf_counter()
  #   interpreter.invoke()
  #   inference_time = time.perf_counter() - start
  #   objs = detect.get_objects(interpreter, args.threshold, scale)
  #   print('%.2f ms' % (inference_time * 1000))

  # print('-------RESULTS--------')
  # if not objs:
  #   print('No objects detected')

  # for obj in objs:
  #   print(labels.get(obj.id, obj.id))
  #   print('  id:    ', obj.id)
  #   print('  score: ', obj.score)
  #   print('  bbox:  ', obj.bbox)

  # if args.output:
  #   image = image.convert('RGB')
  #   draw_objects(ImageDraw.Draw(image), objs, labels)
  #   image.save(args.output)
  #   image.show()


if __name__ == '__main__':
  main()