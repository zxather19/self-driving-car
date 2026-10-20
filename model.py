"""NVIDIA end-to-end steering network (Bojarski et al., 2016).

Layout: 5 convolutional layers -> flatten -> dense 100 -> dense 50 -> dense 10
-> single steering output. ~250k trainable weights.

Written for current Keras 3 / TensorFlow 2.16+ (no legacy layers, no 'mode'
arguments that broke the 2017 example code).
"""

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten
from tensorflow.keras.optimizers import Adam

from utils import IMG_H, IMG_W


def nvidia_model(dropout: float = 0.5, learning_rate: float = 1e-4) -> Sequential:
    model = Sequential(name="nvidia_end_to_end")

    # Convolutional feature extractor: early layers find edges/lane lines,
    # later layers turn them into road shapes.
    model.add(Conv2D(24, (5, 5), strides=(2, 2), activation="elu", input_shape=(IMG_H, IMG_W, 3)))
    model.add(Conv2D(36, (5, 5), strides=(2, 2), activation="elu"))
    model.add(Conv2D(48, (5, 5), strides=(2, 2), activation="elu"))
    model.add(Conv2D(64, (3, 3), activation="elu"))
    model.add(Conv2D(64, (3, 3), activation="elu"))
    model.add(Flatten())

    # Steering regression head. Dropout randomly switches units off during
    # training so the network cannot memorise the training pictures.
    model.add(Dropout(dropout))
    model.add(Dense(100, activation="elu"))
    model.add(Dropout(dropout))
    model.add(Dense(50, activation="elu"))
    model.add(Dropout(dropout))
    model.add(Dense(10, activation="elu"))
    model.add(Dense(1))  # the steering number

    # Mean squared error: big steering misses hurt quadratically more.
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss="mse")
    return model


if __name__ == "__main__":
    m = nvidia_model()
    m.summary()
    print("trainable weights:", m.count_params())
