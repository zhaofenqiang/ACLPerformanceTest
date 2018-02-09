import numpy as np
import os
import tensorflow as tf

def load_graph(model_file):
  graph = tf.Graph()
  graph_def = tf.GraphDef()
  with open(model_file, "rb") as f:
    graph_def.ParseFromString(f.read())
  with graph.as_default():
    tf.import_graph_def(graph_def)
  return graph

graph = load_graph("/home/zfq/ACL1801/ComputeLibrary/scripts/MobileNet/mobilenet_v1_0.50_224.pb")
for t in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
  print(t.name)
