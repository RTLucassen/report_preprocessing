"""
Create predictions for all models on the validation, training, and unlabeled sets.
"""

from utils import ModelPredictor
from pathlib import Path

if __name__ == "__main__":
    
    BASE_DIR = Path().resolve()

    model_path = BASE_DIR / "models"
    # Find all models that are in the model folder
    models = [model.name for model in model_path.iterdir() if model.is_dir()]

    for model_name in models:
        # Predict on the validation set example
        predictor = ModelPredictor(model_name, "validation")
        predictor.predict()

        # Predict on the training set example
        predictor = ModelPredictor(model_name, "training")
        predictor.predict()

        # Predict on the unlabeled set example
        predictor = ModelPredictor(model_name, "unlabeled")
        predictor.predict()