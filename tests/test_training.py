import pandas as pd

from src.training.train_model import build_pipeline


def test_training_pipeline_handles_numeric_and_categorical_features():
    features = pd.DataFrame(
        {
            "age": [25, 35, 45, 55, 65, 30, 40, 50],
            "job": ["admin", "tech", "admin", "services", "retired", "tech", "admin", "services"],
        }
    )
    target = [0, 0, 0, 1, 1, 0, 1, 1]
    pipeline = build_pipeline(features)
    pipeline.fit(features, target)

    probabilities = pipeline.predict_proba(features)
    assert probabilities.shape == (8, 2)
