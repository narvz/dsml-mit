# Deep Learning Cheatsheet

> **Course module:** 5. Deep Learning
> **Code:** [deep_learning.ipynb](deep_learning.ipynb): ANN on tabular data (ExtraaLearn) + CNN on images (Fashion-MNIST), Keras
> **Metrics:** [evaluation_metrics.md § Deep learning](../01_toolkit/evaluation_metrics.md#6-deep-learning-training-curves) + the classification / regression metrics

Deep learning = neural networks with several **hidden layers** that learn their own features from raw data. They shine on **images, text, audio**; on tabular data, tree ensembles are usually as good or better.

---

## The building blocks

| Concept | What it is |
|---|---|
| **Neuron** | `output = activation(w·x + b)`: weighted sum + bias, passed through a non-linear function |
| **Layer** | Many neurons in parallel. **Dense** (fully connected): every input to every neuron |
| **Weights & biases** | The parameters the network learns |
| **Forward pass** | Inputs → layers → prediction |
| **Loss** | How wrong the prediction is (see the table below) |
| **Backpropagation** | Computes the gradient of the loss with respect to every weight (chain rule) |
| **Gradient descent** | Moves the weights a small step against the gradient: `w = w − learning_rate · gradient` |
| **Epoch** | One full pass over the training data |
| **Batch size** | Rows used for each weight update (32–128 typical) |

### Activation functions

| Function | Range | Use |
|---|---|---|
| **ReLU** `max(0, x)` | 0 → ∞ | **Default for hidden layers**. Fast, avoids vanishing gradients |
| **Leaky ReLU** | −∞ → ∞ (small slope < 0) | When ReLU neurons "die" (always output 0) |
| **Sigmoid** | 0 → 1 | **Output of binary classification** |
| **Softmax** | 0 → 1, sums to 1 | **Output of multiclass classification** |
| **Tanh** | −1 → 1 | Hidden layers (older nets, RNNs) |
| **Linear** (none) | −∞ → ∞ | **Output of regression** |

### Output layer + loss: the combination to remember

| Problem | Output layer | Loss |
|---|---|---|
| **Regression** | `Dense(1)` (linear) | `mse` (or `mae`) |
| **Binary classification** | `Dense(1, activation="sigmoid")` | `binary_crossentropy` |
| **Multiclass, integer labels** | `Dense(n_classes, activation="softmax")` | `sparse_categorical_crossentropy` |
| **Multiclass, one-hot labels** (`to_categorical`) | `Dense(n_classes, activation="softmax")` | `categorical_crossentropy` |

### Optimizers

| Optimizer | Notes |
|---|---|
| **SGD** | Plain gradient descent (+ `momentum`). Slow but sometimes generalizes best |
| **RMSprop** | Adapts the learning rate per weight |
| **Adam** | Momentum + RMSprop. **Default choice**, `learning_rate=1e-3` |

**Learning rate** is the most important hyperparameter: too high diverges / jumps around, too low takes forever.

---

## ANN (Artificial Neural Network, a.k.a. MLP)

```python
model = keras.Sequential([
    keras.Input(shape=(n_features,)),
    layers.Dense(64, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(32, activation="relu"),
    layers.Dense(1, activation="sigmoid"),
])
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100, batch_size=32,
          class_weight={0: 1, 1: 2.3}, callbacks=[EarlyStopping(patience=10, restore_best_weights=True)])
```

- **Scale the inputs** (`StandardScaler`). Gradient descent converges badly otherwise.
- Use a **train / validation / test** split: validation drives EarlyStopping, test is used once at the end.
- Imbalanced classes: `class_weight` in `fit()` (same idea as sklearn).
- Start small (1–2 hidden layers, 32–128 neurons), make it bigger only if it underfits.

---

## CNN (Convolutional Neural Network)

| Layer | What it does |
|---|---|
| **Conv2D**(filters, kernel_size) | Slides small filters (e.g. 3×3) over the image. Each filter learns a pattern: edges, then textures, then shapes. **Shares weights**, so far fewer parameters than Dense |
| `padding="same"` | Keeps the output size equal to the input size |
| **MaxPooling2D**(2×2) | Keeps the max of each 2×2 block, halving height & width. Less computation, some shift invariance |
| **BatchNormalization** | Normalizes the activations of each batch, giving faster, more stable training |
| **Flatten** | 3D feature maps → 1D vector for the Dense layers |
| **Dense + Softmax** | Classifier on top of the learned features |

Typical pattern: `[Conv → (BatchNorm) → Pool] × N → Flatten → Dense → Dropout → Softmax`, with **more filters as you go deeper** (32 → 64 → 128).

### Image preprocessing

- **Resize** all images to the same size (`cv2.resize`, `IMG_SIZE = 150`).
- **Normalize** pixels: `X / 255.0` (0–255 → 0–1).
- Shape must be `(n, height, width, channels)`: grayscale = 1 channel, RGB = 3.
- Labels: integers + `sparse_categorical_crossentropy`, or `to_categorical` + `categorical_crossentropy`.
- ⚠️ `cv2.imread` loads images as **BGR**, not RGB. Convert with `cv2.cvtColor(img, cv2.COLOR_BGR2RGB)`.

### Data augmentation

Create modified copies of the training images on the fly: rotation, flips, zoom, shifts. It's the strongest fix for overfitting with few images.

```python
augment = keras.Sequential([layers.RandomFlip("horizontal"), layers.RandomRotation(0.1), layers.RandomZoom(0.1)])
```

### Transfer learning

Instead of training from scratch, reuse a network **pre-trained on ImageNet** (1.4M images):

```python
base = keras.applications.VGG16(weights="imagenet", include_top=False, input_shape=(150, 150, 3))
base.trainable = False                                  # freeze the learned filters
model = keras.Sequential([base, layers.Flatten(), layers.Dense(128, activation="relu"),
                          layers.Dropout(0.3), layers.Dense(n_classes, activation="softmax")])
```

- Works with **little data** and trains fast. The standard approach for real image projects.
- Options: VGG16 (simple, heavy), ResNet50, EfficientNet, MobileNet (light, for mobile).
- **Fine-tuning**: after training the head, unfreeze the last few layers of the base and train with a **very low** learning rate.

---

## Overfitting & regularization

Read the **training vs validation curves**:

| Curves | Diagnosis | Fix |
|---|---|---|
| Both losses high, close together | **Underfitting** | Bigger network, more epochs, higher learning rate |
| Train loss ↓, val loss ↓ then **↑** | **Overfitting** | Regularize (below), EarlyStopping |
| Both low, close together | ✅ Good fit | — |
| Loss jumps around / NaN | Learning rate too high | Lower it |

| Technique | How it helps |
|---|---|
| **EarlyStopping** (`monitor="val_loss"`, `restore_best_weights=True`) | Stops at the best epoch automatically |
| **Dropout** (0.2–0.5) | Randomly drops neurons, so the network can't memorize |
| **L2 / L1** (`kernel_regularizer`) | Penalizes big weights (same idea as Ridge / Lasso) |
| **BatchNormalization** | Stabilizes training, mild regularization |
| **Data augmentation** | More (synthetic) training data |
| **Smaller network** | Less capacity to memorize |

---

## Reproducibility

```python
random.seed(42); np.random.seed(42); tf.random.set_seed(42)   # set in the 3 libraries
keras.backend.clear_session()                                  # free memory between models
```

Training is still slightly random on GPU. Compare models over a few runs.

---

## Common mistakes

- **Not scaling** tabular inputs / not normalizing pixels.
- Wrong **output + loss** pair (e.g. softmax with `binary_crossentropy`, or one-hot labels with `sparse_...`).
- Using the **test set** as validation (EarlyStopping then "sees" the test).
- Training a fixed large number of epochs **without EarlyStopping** and reporting the last epoch.
- Expecting deep learning to beat **XGBoost / Random Forest on small tabular data**. It usually doesn't.
- Training a CNN from scratch on a few thousand real photos instead of using **transfer learning**.
- Forgetting the **channel dimension** (`(28, 28)` instead of `(28, 28, 1)`).

---

## Source notebooks (Drive: `MIT GT - Data Science/5. Deep Learning` + `Hackathon`)

- `Guided Project/Food Recognition.ipynb`: CNN on 150×150 food photos (Bread / Soup / Vegetable-Fruit). Unfinished: it stops after the first conv layers, and the data is a 441 MB zip, not included here
- `Mentor Learning Session/Case_Study - Predicting_Chances_of_Admission.ipynb`: ANN for regression
- `Hackathon/Solution/Shinkansen Travel Experience V3 Ro (ANN 1–3).ipynb`: ANN with Dense + Dropout + L2 + EarlyStopping
