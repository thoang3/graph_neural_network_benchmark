from tensorflow import keras
from matplotlib import pyplot as plt


def run_experiment(model, X_train, y_train, batch_size, epochs, learning_rate):
  optimizer = keras.optimizers.Adam(learning_rate)
  loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
  metrics = [keras.metrics.SparseCategoricalAccuracy(name="accuracy")]
  #metrics = ["accuracy"]

  # Compile the model
  model.compile(
      optimizer=optimizer,
      loss=loss,
      metrics=metrics
  )

  # Callbacks
  callbacks = [
      keras.callbacks.EarlyStopping(
        monitor="val_accuracy", patience=50, restore_best_weights=True)
  ]

  # Fit the model
  history = model.fit(
      x=X_train,
      y=y_train,
      epochs=epochs,
      batch_size=batch_size,
      validation_split=0.15,
      callbacks=callbacks
  )

  return history


def display_learning_curves(history, figure_name=None, display=False):
  fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

  ax1.plot(history.history["loss"])
  ax1.plot(history.history["val_loss"])
  ax1.legend(["train", "test"], loc="upper right")
  ax1.set_xlabel("Epochs")
  ax1.set_ylabel("Loss")

  ax2.plot(history.history["accuracy"])
  ax2.plot(history.history["val_accuracy"])
  ax2.legend(["train", "test"], loc="upper right")
  ax2.set_xlabel("Epochs")
  ax2.set_ylabel("Accuracy")

  if display:
    plt.show()

  if figure_name is None:
    figure_name = "plot.png"
  plt.savefig(figure_name)


