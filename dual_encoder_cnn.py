import tensorflow as tf
import numpy as np
import helpers

FLAGS = tf.flags.FLAGS
DIM = 0
def get_embeddings(hparams):
  initializer = tf.random_uniform_initializer(-0.25, 0.25)

  return tf.get_variable(
    "word_embeddings",
    shape=[hparams.vocab_size, hparams.embedding_dim],
    initializer=initializer)

def get(hparams):
  initializer = tf.random_uniform_initializer(0.5, 0.5)
  return tf.get_variable("drop", shape=[hparams.vocab_size, hparams.embedding_dim],initializer=initializer)

filter_sizes = [2, 4, 5, 7]
num_filters = 64
sequence_length = 175

def dual_encoder_model(hparams, mode, context, context_len, utterance, utterance_len, targets):

  #ok so initialize embeddings
  embeddings_W = get_embeddings(hparams)

  #embedded context and utterance
  context_embedded = tf.nn.embedding_lookup(embeddings_W, context, name="embed_context")
  utterance_embedded = tf.nn.embedding_lookup(embeddings_W, utterance, name="embed_utterance")
  context_embedded_expanded = tf.expand_dims(context_embedded, -1)
  utterance_embedded_expanded = tf.expand_dims(utterance_embedded, -1)
  dropout_keep_prob_utt = tf.Variable(tf.constant(0.5), name="dropout_keep_prob_utt")
  dropout_keep_prob_cont = tf.Variable(tf.constant(0.5), name="dropout_keep_prob_cont")

  #create a convolution + maxpool layer for each  filter size
  pooled_outputs_context = []
  pooled_outputs_utterance = []

  for i, filter_size in enumerate(filter_sizes):
    with tf.name_scope("conv_maxpool_%s" % filter_size):
      #convolution layer
      filter_shape = [filter_size, hparams.embedding_dim, 1, num_filters]
      W_cont = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W_cont")
      b_cont = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b_cont")
      conv_cont = tf.nn.conv2d(context_embedded_expanded, W_cont, strides=[1, 1, 1, 1], padding="VALID", name="conv_cont")

      W_utt = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W_utt")
      b_utt = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b_utt")
      conv_utt = tf.nn.conv2d(utterance_embedded_expanded, W_utt, strides=[1, 1, 1, 1], padding="VALID", name="utt_cont")

      #apply nonlinearity
      h_cont = tf.nn.relu(tf.nn.bias_add(conv_cont, b_cont), name="relu_cont")

      h_utt = tf.nn.relu(tf.nn.bias_add(conv_utt, b_utt), name="relu_utt")

      #maxpooling over outputs
      pooled_cont = tf.nn.max_pool(h_cont, ksize=[1, sequence_length - filter_size + 1, 1, 1 ], strides=[1, 1, 1, 1], padding="VALID", name="pool_cont")
      pooled_outputs_context.append(pooled_cont)

      pooled_utt = tf.nn.max_pool(h_utt, ksize=[1, sequence_length - filter_size + 1, 1, 1 ], strides=[1, 1, 1, 1], padding="VALID", name="pool_utt")
      pooled_outputs_utterance.append(pooled_utt)

  #combine all features
  num_filters_total = num_filters * len(filter_sizes)
  h_pool_cont = tf.concat(3, pooled_outputs_context)
  print ("AVEM",h_pool_cont)
  h_pool_cont_flat = tf.reshape(h_pool_cont, [-1, num_filters_total])
  print("DADADADA",h_pool_cont_flat)
  h_pool_utt = tf.concat(3, pooled_outputs_utterance)
  h_pool_utt_flat = tf.reshape(h_pool_utt, [-1, num_filters_total])

  with tf.name_scope("dropout"):
    DIM = h_pool_utt_flat.get_shape().as_list()[1]
    h_pool_utt_flat = tf.nn.dropout(h_pool_utt_flat, dropout_keep_prob_utt)
    h_pool_cont_flat = tf.nn.dropout(h_pool_cont_flat, dropout_keep_prob_cont)
   # print("DADADADA",h_pool_cont_flat)

  with tf.variable_scope("prediction") as vs:
    M = tf.get_variable("M", shape=[hparams.rnn_dim, hparams.rnn_dim], initializer=tf.truncated_normal_initializer())

    #predict a response c * M
    generated_response = tf.matmul(h_pool_cont_flat, M)
    generated_response = tf.expand_dims(generated_response, 2)
    h_pool_utt_flat = tf.expand_dims(h_pool_utt_flat, 2)

    #dot product between generated response and actual response
    # (c * M)*r
    logits = tf.batch_matmul(generated_response, h_pool_utt_flat, True)
    logits = tf.squeeze(logits, [2])

    if targets is not None:
      print ("TARGETS",targets)
      paddings = [[targets.get_shape().as_list()[0]*8,targets.get_shape().as_list()[0]*7],[0,0]]
      #tf.pad(targets, paddings, "CONSTANT")
      #targets_as_vector = tf.reshape(targets,[-1])
      #zero_padding = tf.zeros([DIM*targets_as_vector.get_shape().as_list()[0]]-tf.shape(targets_as_vector),dtype=targets.dtype)
      #targets_padded = tf.concat(0, [targets_as_vector, zero_padding])
      #targets = tf.reshape(targets_padded, [DIM*targets_as_vector.get_shape().as_list()[0]],1)
      #apply sigmoid to convert logits to probabilities
      #targets = tf.pad(targets,paddings,"CONSTANT" )
      print ("LOGITS",logits)
      print ("TARGETS",targets)

    probs = tf.sigmoid(logits)

    if mode == tf.contrib.learn.ModeKeys.INFER:
      return probs, None

    #calculate the binary cross-entropy loss
    losses = tf.nn.sigmoid_cross_entropy_with_logits(logits, tf.to_float(targets))
  #means loss accross the batch of examples
  mean_loss = tf.reduce_mean(losses, name="mean_loss")
  return probs, mean_loss
