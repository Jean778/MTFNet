import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
from data_handler import read_samples, preprocess_data, generate_envelopes_and_details
from MTFNet import MTFNet

def main():
    CONFIG = {
        "DATA_FOLDER": "Data",
        "FILE_EXTENSION": ".csv",
        "VERBOSE": True,
        "CONCENTRATIONS": [10, 20, 30, 40, 50],
        "CONCENTRATION_DURATION": 960,
        "SMOOTHING_WINDOW": 131,
        "POLYNOMIAL_ORDER": 3,
        "BATCH_SIZE": 32,
        "EPOCHS": 10,
        "LEARNING_RATE": 0.001,
    }

    # Step 1: read data
    file_paths = read_samples(
        data_folder=CONFIG["DATA_FOLDER"],
        verbose=CONFIG["VERBOSE"]
    )

    # Step 2: data preprocess
    X, y_gas, y_concentration = preprocess_data(
        file_path=list(file_paths.values())[0],
        concentrations=CONFIG["CONCENTRATIONS"],
        concentration_duration=CONFIG["CONCENTRATION_DURATION"]
    )
    print(y_gas)
    print(y_concentration)

    # Step 3: generate envelopes and details
    envelopes, details = generate_envelopes_and_details(
        input=X,
        smoothing_window=CONFIG["SMOOTHING_WINDOW"],
        poly_order=CONFIG["POLYNOMIAL_ORDER"]
    )

    # Step 4: data split
    X_train_time, X_val_time, X_train_envelope, X_val_envelope, X_train_detail, X_val_detail, y_train_gas, y_val_gas, y_train_concentration, y_val_concentration = train_test_split(
        X, envelopes, details, y_gas, y_concentration, test_size=0.2
    )

    # Step 5: create and training model
    model = MTFNet((X.shape[1], 1), (envelopes.shape[1],), (details.shape[1],))

    model.fit(
        [X_train_time, X_train_envelope, X_train_detail],
        {"gas_classification_output": y_train_gas, "gas_concentration_output": y_train_concentration},
        validation_data=([X_val_time, X_val_envelope, X_val_detail],
                         {"gas_classification_output": y_val_gas, "gas_concentration_output": y_val_concentration}),
        batch_size=CONFIG["BATCH_SIZE"],
        epochs=CONFIG["EPOCHS"]
    )

    # Step 6: model evaluation
    # accuracy
    y_pred_gas = np.argmax(model.predict([X_val_time, X_val_envelope, X_val_detail])["gas_classification_output"], axis=1)
    acc = accuracy_score(y_val_gas, y_pred_gas)

    # R²
    y_pred_concentration = model.predict([X_val_time, X_val_envelope, X_val_detail])["gas_concentration_output"]
    r2 = r2_score(y_val_concentration, y_pred_concentration)

    # Print evaluation metrics
    print(f"Accuracy: {acc}")
    print(f"R² Score: {r2}")

if __name__ == "__main__":
    main()
