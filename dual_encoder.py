import tensorflow as tf
import numpy as np

FLAGS = tf.flags.FLAGS

def get_embeddings(hparams):
    tf.logging.info("Starting with random embeddings.")
    initializer = tf.random_uniform_initializer(-0.25, 0.25)

    return tf.get_variable("word_embeddings",shape=[hparams.vocab_size, hparams.embedding_dim],initializer=initializer)

def dual_encoder_model(
    hparams,
    mode,
    context,
    context_len,
    utterance,
    utterance_len,
    targets):

  # Initialize embedidngs randomly or with pre-trained vectors if available
  embeddings_W = get_embeddings(hparams)

  # Embed the context and the utterance
  context_embedded = tf.nn.embedding_lookup(
      embeddings_W, context, name="embed_context")
  utterance_embedded = tf.nn.embedding_lookup(
      embeddings_W, utterance, name="embed_utterance")


  # Build the RNN
  with tf.variable_scope("rnn") as vs:
    # We use an LSTM Cell
    cell = tf.nn.rnn_cell.LSTMCell(
        hparams.rnn_dim,
        forget_bias=2.0,
        use_peepholes=True,
        state_is_tuple=True)

    # Run the utterance and context through the RNN
   # rnn_outputs, rnn_states = tf.nn.dynamic_rnn(
   #     cell,
   #     tf.concat(0, [context_embedded, utterance_embedded]),
   #     sequence_length=tf.concat(0, [context_len, utterance_len]),
   #     dtype=tf.float32)
    rnn_outputs, rnn_states = tf.nn.bidirectional_dynamic_rnn(
        cell_fw=cell,
        cell_bw=cell,
        dtype=tf.float32,
        sequence_length=tf.concat(0, [context_len, utterance_len]),
        inputs=tf.concat(0, [context_embedded, utterance_embedded]))
    
    states_fw, states_bw = rnn_states
   # rnn_states = tf.concat(0,[states_fw.h, states_bw.h])
   # rnn_states = tf.truediv(tf.add(states_fw.h,states_bw.h,name=None),2.0,name=None)    
    encoding_context1, encoding_utterance1 = tf.split(0, 2, states_fw.h)
    encoding_context2, encoding_utterance2 = tf.split(0, 2, states_bw.h)
    encoding_context = tf.truediv(tf.add(encoding_context1,encoding_context2,name=None),2.0,name=None)
    encoding_utterance = tf.truediv(tf.add(encoding_utterance1,encoding_utterance2,name=None),2.0,name=None)
  with tf.variable_scope("prediction") as vs:
    M = tf.get_variable("M",
      shape=[hparams.rnn_dim, hparams.rnn_dim],
      initializer=tf.truncated_normal_initializer())

    # "Predict" a  response: c * M
    generated_response = tf.matmul(encoding_context, M)
    generated_response = tf.expand_dims(generated_response, 2)
    encoding_utterance = tf.expand_dims(encoding_utterance, 2)

    # Dot product between generated response and actual response
    # (c * M) * r
    logits = tf.batch_matmul(generated_response, encoding_utterance, True)
    logits = tf.squeeze(logits, [2])

    # Apply sigmoid to convert logits to probabilities
    probs = tf.sigmoid(logits)

    if mode == tf.contrib.learn.ModeKeys.INFER:
      return probs, None

    # Calculate the binary cross-entropy loss
    losses = tf.nn.sigmoid_cross_entropy_with_logits(logits, tf.to_float(targets))

  # Mean loss across the batch of examples
  mean_loss = tf.reduce_mean(losses, name="mean_loss")
  return probs, mean_loss
