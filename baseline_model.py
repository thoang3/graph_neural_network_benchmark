import tensorflow as tf
from tensorflow import keras
from load_cora import load_cora
from utils import run_experiment
from utils import display_learning_curves

layers = tf.keras.layers


# Feed-forward 
def create_ffn(hidden_units, dropout_rate, name=None):
  fnn_layers = []

  for units in hidden_units:
    fnn_layers.append(keras.layers.BatchNormalization())
    fnn_layers.append(keras.layers.Dropout(dropout_rate))
    fnn_layers.append(keras.layers.Dense(units, activation=tf.nn.gelu))

  return keras.Sequential(fnn_layers, name=name)


def create_baseline_model(hidden_units, num_features, num_classes, dropout_rate=0.2):
  inputs = keras.layers.Input(shape=(num_features,), name="input_features")

  x = create_ffn(hidden_units, dropout_rate, name="fnn_block1")(inputs)

  for block_idx in range(6):
    x1 = create_ffn(hidden_units, dropout_rate, name=f"ffn_block{block_idx + 2}")(x)
    # Add skip connection
    x = keras.layers.Add(name=f"skip_connection{block_idx + 2}")([x, x1])

  # Compute logits
  logits = keras.layers.Dense(num_classes, name="logits")(x)

  # Create the model
  return keras.Model(inputs=inputs, outputs=logits, name="baseline")


if __name__ == '__main__':
  papers, train_data, test_data, paper_idx, class_idx, citations, feature_names = load_cora(verbose=1)

  num_features = len(feature_names)
  num_classes = len(class_idx)

  X_train = train_data[feature_names].to_numpy()
  X_test = test_data[feature_names].to_numpy()
  y_train = train_data["subject"]
  y_test = test_data["subject"]

  hidden_units = [32, 32]
  learning_rate = 0.01
  dropout_rate = 0.5
  epochs = 300
  batch_size = 256


  baseline_model = create_baseline_model(hidden_units, num_features, num_classes, dropout_rate)
  baseline_model.summary()


  history = run_experiment(baseline_model, X_train, y_train, batch_size, epochs, learning_rate)

  display_learning_curves(history, figure_name="baseline.png")

  # Evaluate on test data
  _, test_accuracy = baseline_model.evaluate(x=X_test, y=y_test, verbose=1)
  print(test_accuracy)
  print(f"Test accuracy: {round(test_accuracy * 100, 2)}%")
