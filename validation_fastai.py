# validation_fastai_courses.py
import os

# Use the Theano backend if available; the original course was Theano-first.
os.environ.setdefault("KERAS_BACKEND", "theano")

import keras
from keras.models import Sequential
from keras.layers import Dense
import numpy as np

def main():
    # Sanity-check versions
    print("Keras version:", keras.__version__)
    
    # Simple 2-layer MLP to mirror basic course examples
    model = Sequential()
    model.add(Dense(16, input_dim=10, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    # Tiny synthetic dataset
    x = np.random.randn(64, 10).astype("float32")
    y = (np.random.rand(64) > 0.5).astype("float32")

    # One short training run; should complete without error
    history = model.fit(x, y, epochs=1, batch_size=16, verbose=0)
    loss = history.history["loss"][-1]
    print("Final loss:", loss)

    # Basic sanity check: loss should be finite
    assert np.isfinite(loss), "Loss is not finite"

if __name__ == "__main__":
    main()
