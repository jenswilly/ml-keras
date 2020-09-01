# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""label_image for tflite."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import time

import numpy as np
from PIL import Image
# import tensorflow as tf # TF2
import tflite_runtime.interpreter as tflite

# Edge TPU support for current platform
import platform
EDGETPU_SHARED_LIB = {
  'Linux': 'libedgetpu.so.1',
  'Darwin': 'libedgetpu.1.dylib',
  'Windows': 'edgetpu.dll'
}[platform.system()]

def load_labels(filename):
  with open(filename, 'r') as f:
    return [line.strip() for line in f.readlines()]

# From https://stackoverflow.com/a/43357954/1632704
def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '-i',
      '--image',
      default='/tmp/grace_hopper.bmp',
      help='image to be classified')
  parser.add_argument(
      '-m',
      '--model_file',
      default='/tmp/mobilenet_v1_1.0_224_quant.tflite',
      help='.tflite model to be executed')
  parser.add_argument(
      '-c', '--count', type=int, default=5,
      help='Number of times to run inference')
  parser.add_argument( '-t', "--use-edgetpu", type=str2bool, nargs='?',
                        const=True, default=False,
                        help="Activate nice mode.")

  args = parser.parse_args()

  if args.use_edgetpu:
    print( "Using Edge TPU" )
  else:
    print( "Not using Edge TPU" )
  interpreter = tflite.Interpreter( model_path=args.model_file,
                                    experimental_delegates=[tflite.load_delegate( EDGETPU_SHARED_LIB )] if args.use_edgetpu else [] )
  interpreter.allocate_tensors()

  input_details = interpreter.get_input_details()
  output_details = interpreter.get_output_details()

  # check the type of the input tensor
  is_floating_model = input_details[0]['dtype'] == np.float32

  # NxHxWxC, H:1, W:2
  height = input_details[0]['shape'][1]
  width = input_details[0]['shape'][2]
  img = Image.open(args.image).resize((width, height))
  # add N dim
  input_data = np.expand_dims(img, axis=0)

  if is_floating_model:
      input_data = np.float32(input_data) / 255

  interpreter.set_tensor(input_details[0]['index'], input_data)

  first_run = True  # Only print result on first run – we'll get the same results on every invocation
  for i in range( args.count ):
    start_time = time.time()
    interpreter.invoke()
    stop_time = time.time()

    # TODO: Find out what the output format is...
    output_data = interpreter.get_tensor(output_details[0]['index'])
    results = np.squeeze(output_data)

    # JWJ: simpler (binary) classification: 
    if first_run:
      first_run = False
      if results <= 0.02:
        print( f"Cat ({results:.5f})" )
      elif results < 0.98:
        print( f"? ({results:.5f})" )
      else:
        print( f"Dog ({results:.5f})" )

  #   top_k = results.argsort()[-5:][::-1]
  #   labels = load_labels(args.label_file)
  #   for i in top_k:
  #     if floating_model:
  #       print('{:08.6f}: {}'.format(float(results[i]), labels[i]))
  #     else:
  #       print('{:08.6f}: {}'.format(float(results[i] / 255.0), labels[i]))

    print('time: {:.3f}ms'.format((stop_time - start_time) * 1000))

if __name__ == '__main__':
  try:
    main()
  except Exception as e:
    print( f"⚠️  EXCEPTION: {e}" )

