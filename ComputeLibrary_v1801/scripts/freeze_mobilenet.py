#!/usr/bin/python3

from sys import exit
from sys import stdout
from sys import path as syspath
from os import path as osp
from os import stat, makedirs
import argparse

from shutil import copyfile
import tarfile
import json

from six.moves import urllib

import tensorflow as tf
from tensorflow.python.framework import graph_util

parser = argparse.ArgumentParser(
    description='Convert released MobileNetV1 models to frozen .pb format.',
)
parser.add_argument('tf_models_dir',
                    type=str,
                    default=None,
                    help=('The directory created by running '
                          '`git clone '
                          'https://github.com/tensorflow/models.git`.'
                          ),
)
args = parser.parse_args()

if args.tf_models_dir is None:
  print(':(')
  exit()
else:
  slim_dir = osp.join(
      args.tf_models_dir,
      'research',
      'slim',
  )
  syspath.append(slim_dir)

# From tensorflow/models/slim
from nets import mobilenet_v1
from datasets import imagenet

slim = tf.contrib.slim


def download_and_uncompress_tarball(base_url, filename, data_dir):

  def _progress(count, block_size, total_size):
    stdout.write('\r>> Downloading %s %.1f%%' % (
        filename, float(count * block_size) / float(total_size) * 100.0))
    stdout.flush()

  tarball_url = base_url + filename
  filepath = osp.join(data_dir, filename)

  if not tf.gfile.Exists( osp.join(model_dir, model_dl) ):
    filepath, _ = urllib.request.urlretrieve(tarball_url, filepath, _progress)
    print()
    statinfo = stat(filepath)
    print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
  else:
    print('{} tarball already exists -- not downloading'.format(filename))

  tarfile.open(filepath, 'r:*').extractall(data_dir)


def create_label_json_file(json_fn):
  labels = imagenet.create_readable_names_for_imagenet_labels()

  with open(json_fn, 'w') as ofp:
    json.dump(labels, ofp,
              sort_keys=True,
              indent=4,
              separators=(',', ': '))

  return labels


def freeze_mobilenet(meta_file, img_size=224, factor=1.0, num_classes=1001):

  tf.reset_default_graph()

  inp = tf.placeholder(tf.float32,
                      shape=(None, img_size, img_size, 3),
                      name="input")

  is_training=False
  weight_decay = 0.0
  arg_scope = mobilenet_v1.mobilenet_v1_arg_scope(weight_decay=weight_decay)
  with slim.arg_scope(arg_scope):
    logits, _ = mobilenet_v1.mobilenet_v1(inp,
                                          num_classes=num_classes,
                                          is_training=is_training,
                                          depth_multiplier=factor)

  predictions = tf.contrib.layers.softmax(logits)
  output = tf.identity(predictions, name='output')

  ckpt_file = meta_file.replace('.meta', '')
  output_graph_fn = ckpt_file.replace('.ckpt', '.pb')
  output_node_names = "output"

  rest_var = slim.get_variables_to_restore()

  with tf.Session() as sess:
    graph = tf.get_default_graph()
    input_graph_def = graph.as_graph_def()

    saver = tf.train.Saver(rest_var)
    saver.restore(sess, ckpt_file)

    # We use a built-in TF helper to export variables to constant
    output_graph_def = graph_util.convert_variables_to_constants(
        sess, # The session is used to retrieve the weights
        input_graph_def, # The graph_def is used to retrieve the nodes
        # The output node names are used to select the useful nodes
        output_node_names.split(",")
    )

    # Finally we serialize and dump the output graph to the filesystem
    with tf.gfile.GFile(output_graph_fn, "wb") as f:
        f.write(output_graph_def.SerializeToString())
    print("{} ops in the final graph.".format(len(output_graph_def.node)))


factors = ['0.25', '0.50', '0.75', '1.0']
img_sizes = [128, 160, 192, 224]

base_url = 'http://download.tensorflow.org/models/'

model_date = '2017_06_14'
model_base_fmt = 'mobilenet_v1_{}_{}'
model_dl_fmt = model_base_fmt + '_{}.tar.gz'
model_pb_fmt = model_base_fmt + '.pb'

model_dir = './MobileNet'

if not osp.exists(model_dir):
  makedirs(model_dir)

json_fn = osp.join(model_dir, 'labels.json')

labels = create_label_json_file(json_fn)
num_classes = len(labels)

if not tf.gfile.Exists(model_dir):
  tf.gfile.MakeDirs(model_dir)

for img_size in img_sizes:
  img_subdir = osp.join(model_dir, 'img{}'.format(img_size))
  if not tf.gfile.Exists(img_subdir):
    tf.gfile.MakeDirs(img_subdir)

  for factor in factors:
    model_dl = model_dl_fmt.format(factor, img_size, model_date)
    model_pb = model_pb_fmt.format(factor, img_size)

    if not tf.gfile.Exists( osp.join(img_subdir, model_pb) ):

      download_and_uncompress_tarball(base_url, model_dl, model_dir)

      try:
        meta_file = osp.join(
            model_dir,
            model_pb.replace('.pb'.format(model_date), '.ckpt.meta'),
        )
        if tf.gfile.Exists( meta_file ):
          print('Processing meta_file {}'.format(meta_file))
          freeze_mobilenet(meta_file, img_size, float(factor), num_classes)
          copyfile(osp.join(model_dir, model_pb),
                   osp.join(img_subdir, model_pb))
        else:
          print('Skipping meta file {}'.format(meta_file))
          pass
      except:
        print('Failed to process meta_file {}'.format(meta_file))
    else:
      print('{} frozen model already exists -- skipping'.format(model_pb))
