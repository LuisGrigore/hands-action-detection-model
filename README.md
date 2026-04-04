# Hand Action Detection Model

A machine learning project for real-time hand gesture recognition using MediaPipe hand landmarks and deep learning models.

## Features

- **Data Collection**: Capture hand landmark sequences from webcam for multiple gestures
- **Model Training**: Train GRU-based neural networks on collected datasets
- **Real-Time Testing**: Evaluate model performance with live video feed and gesture prediction
- **Configurable**: Flexible configuration for hand selection, sequence length, and model architecture

## Installation

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Set up Python environment (recommended: virtualenv or conda)

## Usage

1. **Collect Data**: Run `collect_data/collect_data.ipynb` to record gesture sequences
2. **Train Model**: Execute `train_model/train_model.ipynb` to train the recognition model
3. **Test Model**: Use `test_model/test_model.ipynb` for real-time gesture detection

## Configuration

Edit `config.json` to customize:
- Gesture labels
- Sequence parameters
- Model architecture
- Training hyperparameters

## Requirements

- Python 3.8+
- TensorFlow/Keras
- OpenCV
- MediaPipe
- NumPy, scikit-learn

## License

See LICENSE file for details.