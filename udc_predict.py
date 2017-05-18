import os
import time
import itertools
import sys
import numpy as np
import tensorflow as tf
import udc_model
import udc_hparams
import udc_metrics
import udc_inputs
#from dual_encoder import dual_encoder_model
from dual_encoder_cnn import dual_encoder_model
from helpers import load_vocab
from operator import itemgetter

tf.logging.set_verbosity(tf.logging.ERROR)
tf.flags.DEFINE_string("model_dir", None, "Directory to load model checkpoints from")
tf.flags.DEFINE_string("vocab_processor_file", "./data/vocab_processor.bin", "Saved vocabulary processor file")
FLAGS = tf.flags.FLAGS

if not FLAGS.model_dir:
  print("You must specify a model directory")
  sys.exit(1)

def tokenizer_fn(iterator):
  return (x.split(" ") for x in iterator)

# Load vocabulary
vp = tf.contrib.learn.preprocessing.VocabularyProcessor.restore(
  FLAGS.vocab_processor_file)

# Load your own data here
INPUT_CONTEXT = "do you love me" #"can you make it at time at 4 pm"  #"i want to break up with you" #"where do you want to get lunch"
POTENTIAL_RESPONSES_NEGATIVE = ["i am bored","i am working at home","no i don t want that","i d hate that","not hungry right now","i can t reach the meeting can we meet later","babe i m gonna leave you"]
POTENTIAL_RESPONSES_POSITIVE = ["i am at the gym","i d love to","let s eat at an italian restaurant", "i love you babe","i will try to reach the meeting"]

def get_features(context, utterance):
  context_matrix = np.array(list(vp.transform([context])))
  utterance_matrix = np.array(list(vp.transform([utterance])))
  context_len = len(context.split(" "))
  utterance_len = len(utterance.split(" "))
  features = {
    "context": tf.convert_to_tensor(context_matrix, dtype=tf.int64),
    "context_len": tf.constant(context_len, shape=[1,1], dtype=tf.int64),
    "utterance": tf.convert_to_tensor(utterance_matrix, dtype=tf.int64),
    "utterance_len": tf.constant(utterance_len, shape=[1,1], dtype=tf.int64),
  }
  return features, None

if __name__ == "__main__":
  hparams = udc_hparams.create_hparams()
  model_fn = udc_model.create_model_fn(hparams, model_impl=dual_encoder_model)
  estimator = tf.contrib.learn.Estimator(model_fn=model_fn, model_dir=FLAGS.model_dir)

  # Ugly hack, seems to be a bug in Tensorflow
  # estimator.predict doesn't work without this line
  estimator._targets_info = tf.contrib.learn.estimators.tensor_signature.TensorSignature(tf.constant(0, shape=[1,1]))

  probs_p = []
  probs_n = []
  #print("Context: {}".format(INPUT_CONTEXT))
  for r in POTENTIAL_RESPONSES_POSITIVE:
    prob = estimator.predict(input_fn=lambda: get_features(INPUT_CONTEXT, r))
    probs_p.append((r,prob))
    #print("{}: {:g}".format(r, prob[0,0]))
  for r in POTENTIAL_RESPONSES_NEGATIVE:
    prob = estimator.predict(input_fn=lambda: get_features(INPUT_CONTEXT, r))
    probs_n.append((r,prob))

  print (max(probs_p,key=itemgetter(1))[0])
  print (max(probs_n,key=itemgetter(1))[0])
