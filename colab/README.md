# TSL-51 Colab Notebooks

This directory contains Google Colab notebooks for training and evaluating Thai Sign Language recognition models.

## Notebooks

### 01_setup.ipynb
**Purpose**: Environment setup and dependency installation

**What it does**:
- Mounts Google Drive
- Installs PyTorch and other dependencies
- Verifies GPU availability
- Tests module imports
- Creates output directories (models/, results/, data/)

**When to run**: First time using Colab or when dependencies need updating

### 02_train.ipynb
**Purpose**: Train TSL-51 models with K-Fold cross-validation

**What it does**:
- Loads dataset from HuggingFace
- Creates model (GRU or MLP)
- Performs K-Fold cross-validation
- Saves best model checkpoint
- Generates training results

**Configuration**:
```python
CONFIG = {
    'dataset': 'tsl51_user_sign',  # Dataset choice
    'model_type': 'gru',           # Model architecture
    'hidden_dim': 256,             # Hidden layer size
    'num_layers': 3,               # Number of layers
    'dropout': 0.3,                # Dropout rate
    'learning_rate': 0.001,        # Learning rate
    'batch_size': 64,              # Batch size
    'epochs': 50,                 # Training epochs
    'k_folds': 5,                 # K-Fold CV
    'augment_factor': 2,          # Data augmentation
    'early_stopping_patience': 10 # Early stopping
}
```

**When to run**: After setup, when you want to train a new model

### 03_evaluate.ipynb
**Purpose**: Evaluate trained models on test set

**What it does**:
- Loads trained model checkpoint
- Splits data into train/test
- Evaluates on test set
- Generates confusion matrix
- Calculates per-class accuracy
- Saves evaluation results

**When to run**: After training to assess model performance

### 04_inference.ipynb
**Purpose**: Run inference on video files

**What it does**:
- Loads trained model
- Initializes MediaPipe for landmark extraction
- Uploads video file
- Extracts features from video frames
- Runs model inference
- Displays prediction timeline and statistics

**When to run**: After training to test on new videos

## Usage Workflow

1. **Run 01_setup.ipynb** - Set up environment
2. **Run 02_train.ipynb** - Train your model
3. **Run 03_evaluate.ipynb** - Evaluate performance
4. **Run 04_inference.ipynb** - Test on videos

## Tips

- **GPU Usage**: Colab notebooks automatically use GPU when available. Free tier has limited runtime.
- **Google Drive**: All models and results are saved to Google Drive for persistence.
- **Dataset Caching**: First dataset download takes time but is cached for subsequent runs.
- **Model Paths**: Update `MODEL_PATH` in notebooks to point to your trained model.
- **Configuration**: Modify the `CONFIG` dictionary in 02_train.ipynb to experiment with different settings.

## Troubleshooting

**Out of Memory**: Reduce `batch_size` or `hidden_dim` in configuration.

**Slow Training**: Reduce `epochs` or `k_folds` for faster iteration.

**Import Errors**: Ensure 01_setup.ipynb has been run successfully first.

**GPU Not Available**: Colab may not have GPU available; check runtime type in Colab menu.
