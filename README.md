# Neural Networks

A collection of neural network examples demonstrating machine learning concepts using TensorFlow/Keras, including binary classification, image convolution, and handwritten digit recognition.

## Project Structure

```
neural_networks/
├── banknotes/           # Binary classification for banknote authenticity
│   ├── banknotes.py     # Neural network training script
│   └── banknotes.csv    # Dataset with banknote features
├── convolution/         # Image filtering demonstrations
│   ├── filter.py        # Edge detection using convolution kernels
│   └── bridge.png       # Sample image for filtering
├── digits/              # Handwritten digit recognition (MNIST)
│   ├── handwriting.py   # CNN training script
│   ├── recognition.py   # Interactive GUI for digit recognition
│   ├── model.keras      # Trained model (generated after training)
│   └── assets/
│       └── fonts/       # Fonts for the pygame GUI
└── README.md
```

## Requirements

### Python Dependencies

| Package        | Purpose                                  |
|----------------|------------------------------------------|
| `tensorflow`   | Deep learning framework (includes Keras) |
| `numpy`        | Numerical computing                      |
| `scikit-learn` | Data splitting utilities                 |
| `pygame`       | Interactive GUI for digit recognition    |
| `Pillow`       | Image processing for convolution         |

### System Requirements

- **Python Version**: 3.11 or 3.12 recommended
  - ⚠️ Python 3.14 may have compatibility issues (no pre-built wheels for pygame/tensorflow)
- **macOS Users**: SDL2 libraries required for pygame (see Installation)

## Installation

### 1. Create Virtual Environment

```bash
# Using Python 3.11 or 3.12
python3.11 -m venv venv
source venv/bin/activate  # On macOS/Linux
# or
venv\Scripts\activate     # On Windows
```

### 2. Install Dependencies

```bash
pip install tensorflow numpy scikit-learn pygame Pillow
```

### macOS: Installing SDL Dependencies (if pygame fails to build)

If you encounter `SDL.h file not found` errors:

```bash
# Install SDL2 libraries via Homebrew
brew install sdl2 sdl2_image sdl2_mixer sdl2_ttf

# Then install pygame
pip install pygame
```

## Usage

### Banknotes - Binary Classification

Classifies banknotes as authentic or counterfeit based on image features (variance, skewness, curtosis, entropy).

```bash
cd banknotes
python banknotes.py
```

**Output**: Trains a simple neural network and displays accuracy metrics.

**Dataset**: `banknotes.csv` contains 1,372 samples with 4 features each.

---

### Convolution - Image Filtering

Demonstrates edge detection using a 3x3 convolution kernel.

```bash
cd convolution
python filter.py bridge.png
```

**Output**: Opens a window displaying the edge-detected version of the input image.

**Kernel Used**: Laplacian edge detection `[-1, -1, -1, -1, 8, -1, -1, -1, -1]`

---

### Digits - Handwritten Digit Recognition

Interactive application for recognizing handwritten digits (0-9) using a Convolutional Neural Network trained on MNIST.

#### Step 1: Train the Model

```bash
cd digits
python handwriting.py model.keras
```

Or run without arguments to use the default filename:

```bash
python handwriting.py
# Saves to model.keras automatically
```

**Training Details**:
- Dataset: MNIST (60,000 training images, 10,000 test images)
- Architecture: Conv2D → MaxPooling → Dense → Dropout → Softmax
- Epochs: 10
- Expected accuracy: ~98-99%

#### Step 2: Run the Recognition GUI

```bash
python recognition.py model.keras
```

**GUI Controls**:
- **Draw**: Click and drag on the 28×28 grid to draw a digit
- **Classify**: Click to predict the drawn digit
- **Reset**: Clear the drawing canvas

## Data Files

| File                      | Description                                                               |
|---------------------------|---------------------------------------------------------------------------|
| `banknotes/banknotes.csv` | Banknote authentication dataset (1,372 samples, 4 features + class label) |
| `convolution/bridge.png`  | Sample image for edge detection demonstration                             |
| `digits/model.keras`      | Trained CNN model (generated after running `handwriting.py`)              |

### MNIST Dataset

TensorFlow automatically downloads the MNIST dataset on first run of `handwriting.py`. It contains:
- 60,000 training images of handwritten digits
- 10,000 test images
- Each image is 28×28 pixels, grayscale

## Model Format

This project uses the **native Keras format** (`.keras` extension) for saving models.

```python
model.save("model.keras")  # Recommended
```

> ⚠️ The legacy HDF5 format (`.h5`) is deprecated. If you have old `.h5` model files, retrain and save using `.keras`.

## Notes & Tips

### Type Annotations
The code uses Python type hints for better IDE support and to avoid type checker warnings:
```python
handwriting: list[list[float]] = [[0.0] * COLS for _ in range(ROWS)]
```

### NumPy Array Conversion
Keras requires NumPy arrays, not Python lists. Always convert before training:
```python
X_training = np.array(X_training)
y_training = np.array(y_training)
```

### Input Layer Best Practice
Use explicit `Input` layers instead of `input_shape` parameters to avoid deprecation warnings:
```python
# Recommended
tf.keras.layers.Input(shape=(28, 28, 1)),
tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),

# Deprecated
tf.keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1))
```

## License

This project is for educational purposes.

