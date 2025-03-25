import pytest
import pandas as pd
import numpy as np
from prediction_demo import data_preparation, data_split, train_model, eval_model

@pytest.fixture
def housing_data_sample():
    return pd.DataFrame(
        data={
            'price': [13300000, 12250000],
            'area': [7420, 8960],
            'bedrooms': [4, 4],    
            'bathrooms': [2, 4],    
            'stories': [3, 4],    
            'mainroad': ["yes", "yes"],    
            'guestroom': ["no", "no"],    
            'basement': ["no", "no"],    
            'hotwaterheating': ["no", "no"],    
            'airconditioning': ["yes", "yes"],    
            'parking': [2, 3],
            'prefarea': ["yes", "no"],    
            'furnishingstatus': ["furnished", "unfurnished"]
        }
    )

def test_data_preparation(housing_data_sample):
    feature_df, target_series = data_preparation(housing_data_sample)
    
    # Target and datapoints have the same length
    assert feature_df.shape[0] == len(target_series)

    # Feature only has numerical values
    assert feature_df.shape[1] == feature_df.select_dtypes(include=(np.number, np.bool_)).shape[1]

@pytest.fixture
def feature_target_sample(housing_data_sample):
    feature_df, target_series = data_preparation(housing_data_sample)
    return (feature_df, target_series)

def test_data_split(feature_target_sample):
    return_tuple = data_split(*feature_target_sample)
    
    # Verificar que la función devuelve exactamente 4 elementos (X_train, X_test, y_train, y_test)
    assert len(return_tuple) == 4, "data_split should return 4 elements"

    X_train, X_test, y_train, y_test = return_tuple

    # Verificar que los tamaños de los conjuntos de entrenamiento y prueba son correctos
    total_samples = len(feature_target_sample[0])
    expected_train_size = int(0.8 * total_samples)  # Asumiendo una división 80/20
    expected_test_size = total_samples - expected_train_size

    assert len(X_train) == expected_train_size, f"Expected {expected_train_size} train samples, got {len(X_train)}"
    assert len(X_test) == expected_test_size, f"Expected {expected_test_size} test samples, got {len(X_test)}"
    assert len(y_train) == expected_train_size, f"Expected {expected_train_size} train labels, got {len(y_train)}"
    assert len(y_test) == expected_test_size, f"Expected {expected_test_size} test labels, got {len(y_test)}"