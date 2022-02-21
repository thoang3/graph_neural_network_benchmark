import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from load_cora import load_cora
from baseline_model import create_ffn
from utils import run_experiment
from utils import display_learning_curves


# Graph convolution layer
class GraphConvLayer(layers.Layer):
  def __init__(
      self,
      hidden_units,
      dropout_rate=0.2,
      aggregation_type="mean",
      combination_type="concat",
      normalize=False,
      *args,
      **kwargs
  ):
    super(GraphConvLayer, self).__init__(*args, **kwargs)

    self._aggregation_type = aggregation_type
    self._combination_type = combination_type
    self._normalize = normalize

    self._ffn_prepare = create_ffn(hidden_units, dropout_rate)
    if self._combination_type == "gated":
      self._update_fn = layers.GRU(
          units=hidden_units,
          activation="tanh",
          recurrent_activation="sigmoid",
          dropout=dropout_rate,
          return_state=True,
          recurrent_dropout=dropout_rate
      )
    else:
      self._update_fn = create_ffn(hidden_units, dropout_rate)

  def _prepare(self, node_representations, weights=None):
    # node_representations shape is [num_edges, embedding_dim]
    messages = self._ffn_prepare(node_representations)
    if weights is not None:
      messages = messages * tf.expand_dims(weights, -1)

    return messages

  def _aggregate(self, node_indices, neighbour_messages):
    # node_indices shape is [num_edges]
    # neighbour_messages shape: [num_edges, representation_dim]
    num_nodes = tf.math.reduce_max(node_indices) + 1
    if self._aggregation_type == "sum":
      aggregated_message = tf.math.unsorted_segment_sum(
          neighbour_messages,
          node_indices,
          num_segments=num_nodes
      )
    elif self._aggregation_type == "mean":
      aggregated_message = tf.math.unsorted_segment_mean(
          neighbour_messages,
          node_indices,
          num_segments=num_nodes
      )
    elif self._aggregation_type == "max":
      aggregated_message = tf.math.unsorted_segment_max(
          neighbour_messages,
          node_indices,
          num_segments=num_nodes
      )
    else:
      raise ValueError(f"Invalid aggregation type: {self._aggregation_type}.")

    return aggregated_message

  def _update(self, node_representations, aggregated_messages):
    # node_representations shape is [num_nodes, representation_dim]
    # aggregated_messages shape is [num_nodes, representation_dim]
    if self._combination_type == "gru":
      # Create a sequence of two elements for the GRU layer
      h = tf.stack([node_respresentations, aggregated_messages], axis=1)
    elif self._combination_type == "concat":
      # Concatenate the node_representations and aggregated_messages
      h = tf.concat([node_representations, aggregated_messages], axis=1)
    elif self._combination_type == "add":
      # Add node_representations and aggregated_messages
      h = node_representations + aggregated_messages
    else:
      raise ValueError(f"Invalid combination type: {self._combinatino_type}.")

    # Apply the processing function
    node_embeddings = self._update_fn(h)
    if self._combination_type == "gru":
      node_embeddings = tf.unstack(node_embeddings, axis=1)[-1]

    if self._normalize:
      node_embeddings = tf.nn.l2_normalize(node_embeddings, axis=-1)

    return node_embeddings

  def call(self, inputs):
    """Process the inputs to produce the node_embeddings.

    Args:
      Inputs:
        A tuple of three elements: node_representations, edges, edge_weights.
    Returns:
      node_embeddings of shape [num_nodes, representation_dim].
    """

    node_representations, edges, edge_weights = inputs
    # Get node_indices (source) and neighbour_indices (target) from edges
    node_indices, neighbour_indices = edges[0], edges[1]
    # neighbour_representations shape is [num_edges, representation_dim]
    neighbour_representations = tf.gather(node_representations, neighbour_indices)

    # Prepare the messages of the neighbours
    neighbour_messages = self._prepare(neighbour_representations, edge_weights)
    # Aggregate the neighbour messages
    aggregated_messages = self._aggregate(node_indices, neighbour_messages)

    # Update the node embedding with the neighbour messages
    return self._update(node_representations, aggregated_messages)


class GNNNodeClassifier(tf.keras.Model):
  def __init__(
      self,
      graph_info,
      num_classes,
      hidden_units,
      aggregation_type="sum",
      combination_type="concat",
      dropout_rate=0.2,
      normalize=True,
      *args,
      **kwargs
  ):
    super(GNNNodeClassifier, self).__init__(*args, **kwargs)

    # Unpack graph_info
    node_features, edges, edge_weights = graph_info
    self._node_features = node_features
    self._edges = edges
    self._edge_weights = edge_weights
    # Set edge_weights to ones if not provided
    if self._edge_weights is None:
      self._edge_weights = tf.ones(shape=edges.shape[1])
    # Scale edge_weights to sum to 1
    self._edge_weights = self._edge_weights / tf.math.reduce_sum(self._edge_weights)

    # Create a process layer
    self._preprocess = create_ffn(hidden_units, dropout_rate, name="preprocess")
    # Create the 1st GraphConv layer
    self._conv1 = GraphConvLayer(
        hidden_units,
        dropout_rate,
        aggregation_type,
        combination_type,
        normalize,
        name="graph_conv1"
    )
    # Create the 2nd GraphConv layer
    self._conv2 = GraphConvLayer(
        hidden_units,
        dropout_rate,
        aggregation_type,
        combination_type,
        normalize,
        name="graph_conv2"
    )
    # Create a postprocess layer
    self._postprocess = create_ffn(hidden_units, dropout_rate, name="postprocess")
    # Create a compute logits layer
    self._compute_logits = layers.Dense(units=num_classes, name="logits")

  def call(self, input_node_indices):
    # Preprocess the node_features to produce node representations
    x = self._preprocess(self._node_features)
    # Apply the 1st graph conv layer
    x1 = self._conv1((x, self._edges, self._edge_weights))
    # Skip connection
    x = x1 + x
    # Apply the 2nd graph conv layer
    x2 = self._conv2((x, self._edges, self._edge_weights))
    # Skip connection
    x = x2 + x
    # Postprocess node embedding
    x = self._postprocess(x)
    # Fetch node embeddings for the input node_indices
    node_embeddings = tf.gather(x, input_node_indices)

    # Compute logits
    return self._compute_logits(node_embeddings)


if __name__ == '__main__':

  papers, train_data, test_data, paper_idx, class_idx, citations, feature_names  = load_cora(verbose=1)

  num_features = len(feature_names)
  num_classes = len(class_idx)

  hidden_units = [32, 32]
  learning_rate = 0.01
  dropout_rate = 0.5
  epochs = 300
  batch_size = 256

  # Create an edges array (sparse adjacency matrix) of shape [2, num_edges]
  edges = citations[["source", "target"]].to_numpy().T
  #print(edges)
  # Create an edge weights array of ones (default weights)
  edge_weights = tf.ones(shape=edges.shape[1])
  # Create a node features array of shape [num_nodes, num_features]
  node_features = tf.cast(
      papers.sort_values("paper_id")[feature_names].to_numpy(), dtype=tf.float32)

  # Create graph info tuple with node_features, edges, and edge_weights
  graph_info = (node_features, edges, edge_weights)

  print("Edges shape: ", edges.shape)
  print("Nodes shape: ", node_features.shape)

  gnn_model = GNNNodeClassifier(
      graph_info=graph_info,
      num_classes=num_classes,
      hidden_units=hidden_units,
      dropout_rate=dropout_rate,
      name="gnn_model"
  )

  print("GNN output shape: ", gnn_model([1, 10, 100]))

  gnn_model.summary()

  # Train the GNN model
  X_train = train_data.paper_id.to_numpy()
  y_train = train_data.subject
  history = run_experiment(gnn_model, X_train, y_train, batch_size, epochs, learning_rate)

  # Plot the learning curves
  display_learning_curves(history, figure_name="gnn.png")

  # Evaluate on test data
  X_test = test_data.paper_id.to_numpy()
  y_test = test_data.subject
  _, test_accuracy = gnn_model.evaluate(x=X_test, y=y_test, verbose=1)
  print(f"Test accuracy: {round(test_accuracy * 100, 2)}%")


