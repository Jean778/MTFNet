import os
import numpy as np
import pandas as pd
from scipy.signal import hilbert, savgol_filter
from sklearn.preprocessing import LabelEncoder

CONFIG = {
    "DATA_FOLDER": "Data",
    "FILE_EXTENSION": ".csv",
    "VERBOSE": True,
    "FILE_PATH": None,
    "CONCENTRATIONS": [],
    "CONCENTRATION_DURATION": None
}

def read_samples(data_folder: str, verbose: bool) -> dict:
    """
    Reads sample files from the specified folder.

    Parameters:
        data_folder (str): Path to the folder containing data files.
        verbose (bool): Whether to print detailed logs.

    Returns:
        dict: A dictionary mapping file names (without extension) to their file paths.
    """
    file_path = {}
    for filename in os.listdir(data_folder):
        filepath = os.path.join(data_folder, filename)
        try:
            file_path[filename] = filepath
            if verbose:
                print(f"Found file: {filename} at {filepath}")
        except Exception as e:
            if verbose:
                print(f"Failed to process file: {filename}, Error: {e}")
    return file_path

def preprocess_data(file_path: str, concentrations: list, concentration_duration: int):

    data = pd.read_excel(file_path)
    columns = data.columns[1:]
    X = []
    y_gas = []
    y_concentration = []

    for col in columns:
        gas_name = col.split('_')[0]
        for i, conc in enumerate(concentrations):
            start_index = i * concentration_duration
            end_index = start_index + concentration_duration
            data_segment = data[col].iloc[start_index:end_index].values
            X.append(data_segment)
            y_gas.append(gas_name)
            y_concentration.append(conc)

    X = np.array(X)
    y_gas = np.array(y_gas)
    y_concentration = np.array(y_concentration)
    label_encoder = LabelEncoder()
    y_gas = label_encoder.fit_transform(y_gas)
    y_concentration = label_encoder.fit_transform(y_concentration)

    return X, y_gas, y_concentration


# Compute the envelope of a signal
def compute_envelope(input):

    analytic_signal = hilbert(input)
    envelope = np.abs(analytic_signal)
    return envelope

# Generate envelopes and detail signals
def generate_envelopes_and_details(input, smoothing_window: int, poly_order: int):

    envelopes = []
    details = []
    for sample in input:
        envelope = compute_envelope(sample)
        smoothed_envelope = savgol_filter(envelope, smoothing_window, poly_order)
        detail = sample - smoothed_envelope
        envelopes.append(smoothed_envelope)
        details.append(detail)
    return np.array(envelopes), np.array(details)

