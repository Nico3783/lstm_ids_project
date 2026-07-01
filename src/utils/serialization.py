
# src/utils/serialization.py
# Project: Deep Learning IDS Using LSTM
# Developer: Kayode Timileyin Nicholas
# Purpose: Unified save and load functions for every artifact
#          the pipeline produces — Keras models (.keras, .h5),
#          scikit-learn objects (scaler, label encoder,
#          baseline models) via pickle/joblib, NumPy arrays
#          (.npy), and JSON metadata files.
#
#          Having one serialization module guarantees that
#          the training, evaluation, and deployment modules
#          all read and write artifacts in exactly the same
#          format and location, preventing version mismatches
#          between the model and its preprocessing artifacts.

import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np

from src.utils.logger import get_logger
from src.utils.helpers import _json_serialiser

logger = get_logger(__name__)


# Keras / TensorFlow Model Serialization

def save_keras_model(
    model: Any,
    path: Union[str, Path],
    save_h5_copy: bool = True,
) -> None:
    """
    Save a compiled Keras model to disk in the native ``.keras``
    format and optionally as a legacy ``.h5`` file.

    Both formats are saved so the project satisfies the
    ``models/final/`` structure defined in PROJECT_TREE.txt
    (``lstm_ids_model.keras`` and ``lstm_ids_model.h5``).

    Parameters
    ----------
    model : keras.Model
        Compiled and trained Keras model instance.
    path : str or Path
        Destination path.  Must end with ``.keras``.
        The ``.h5`` copy (if requested) is saved to the same
        directory with the same stem but ``.h5`` extension.
    save_h5_copy : bool
        When True, also save a ``.h5`` copy.  Defaults to True.

    Raises
    ------
    ValueError
        If *path* does not end with ``.keras``.
    """
    path = Path(path)
    if path.suffix != ".keras":
        raise ValueError(
            f"Keras model path must end with '.keras', got: {path}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)

    model.save(str(path))
    logger.info("Keras model saved (.keras): %s", path)

    if save_h5_copy:
        h5_path = path.with_suffix(".h5")
        model.save(str(h5_path))
        logger.info("Keras model saved (.h5 copy): %s", h5_path)


def load_keras_model(
    path: Union[str, Path],
    custom_objects: Optional[Dict[str, Any]] = None,
) -> Any:
    """
    Load a Keras model from a ``.keras`` or ``.h5`` file.

    Parameters
    ----------
    path : str or Path
        Path to the saved model file.
    custom_objects : dict, optional
        Dictionary mapping names to custom classes or
        functions required to deserialise the model (e.g.
        custom loss functions or metrics).

    Returns
    -------
    keras.Model
        Loaded and ready-to-use Keras model.

    Raises
    ------
    FileNotFoundError
        If the model file does not exist.
    """
    import tensorflow as tf  # type: ignore

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Keras model file not found: {path}\n"
            "Run the training pipeline first: python train.py"
        )
    logger.info("Loading Keras model from: %s", path)
    model = tf.keras.models.load_model(
        str(path), custom_objects=custom_objects
    )
    logger.info(
        "Keras model loaded successfully. Parameters: {:,}".format(
            model.count_params()
        )
    )
    return model


# Scikit-learn / Joblib Serialization
# (Scaler, LabelEncoder, Baseline Models)

def save_object(
    obj: Any,
    path: Union[str, Path],
    use_joblib: bool = True,
) -> None:
    """
    Serialise any Python object (scaler, label encoder,
    baseline model) to disk using joblib or pickle.

    Joblib is preferred for NumPy-heavy objects (e.g.
    RandomForest) because it serialises large arrays more
    efficiently than pickle.

    Parameters
    ----------
    obj : Any
        Python object to serialise.  Typically a scikit-learn
        estimator, ``MinMaxScaler``, or ``LabelEncoder``.
    path : str or Path
        Destination ``.pkl`` file path.
    use_joblib : bool
        Use ``joblib.dump`` when True (default), else ``pickle``.

    Examples
    --------
    >>> save_object(scaler, paths.SCALER_FINAL_PATH)
    >>> save_object(label_encoder, paths.LABEL_ENCODER_FINAL_PATH)
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if use_joblib:
        joblib.dump(obj, str(path))
    else:
        with open(path, "wb") as fh:
            pickle.dump(obj, fh, protocol=pickle.HIGHEST_PROTOCOL)

    logger.info(
        "Object saved (%s): %s",
        type(obj).__name__,
        path,
    )


def load_object(
    path: Union[str, Path],
    use_joblib: bool = True,
) -> Any:
    """
    Deserialise a Python object from a ``.pkl`` file.

    Parameters
    ----------
    path : str or Path
        Source ``.pkl`` file path.
    use_joblib : bool
        Use ``joblib.load`` when True (default), else ``pickle``.

    Returns
    -------
    Any
        The deserialised Python object.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Serialised object not found: {path}\n"
            "Ensure the preprocessing pipeline has been run."
        )

    if use_joblib:
        obj = joblib.load(str(path))
    else:
        with open(path, "rb") as fh:
            obj = pickle.load(fh)

    logger.info(
        "Object loaded (%s): %s",
        type(obj).__name__,
        path,
    )
    return obj


# NumPy Array Serialization

def save_numpy_array(array: np.ndarray, path: Union[str, Path]) -> None:
    """
    Save a NumPy array to a ``.npy`` binary file.

    Parameters
    ----------
    array : np.ndarray
        Array to save.
    path : str or Path
        Destination ``.npy`` file path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(path), array)
    logger.debug(
        "NumPy array saved — shape %s, dtype %s: %s",
        array.shape,
        array.dtype,
        path,
    )


def load_numpy_array(path: Union[str, Path]) -> np.ndarray:
    """
    Load a NumPy array from a ``.npy`` file.

    Parameters
    ----------
    path : str or Path
        Source ``.npy`` file path.

    Returns
    -------
    np.ndarray
        Loaded array.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"NumPy array file not found: {path}\n"
            "Run the preprocessing pipeline first: python run_pipeline.py"
        )
    array = np.load(str(path), allow_pickle=False)
    logger.debug(
        "NumPy array loaded — shape %s, dtype %s: %s",
        array.shape,
        array.dtype,
        path,
    )
    return array


def save_processed_arrays(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    output_dir: Union[str, Path],
) -> None:
    """
    Save all six processed split arrays to *output_dir* using
    the standard filenames defined in ``constants.py``.

    This function is called once by the preprocessing pipeline
    after sequence building and splitting are complete, writing
    the final inputs that the training module will load.

    Parameters
    ----------
    X_train, X_val, X_test : np.ndarray
        3-D feature arrays of shape
        ``(n_samples, window_size, n_features)``.
    y_train, y_val, y_test : np.ndarray
        1-D integer label arrays of shape ``(n_samples,)``.
    output_dir : str or Path
        Directory where the ``.npy`` files will be written
        (typically ``data/processed/``).
    """
    from src.utils.constants import (
        X_TRAIN_NPY, X_VAL_NPY, X_TEST_NPY,
        Y_TRAIN_NPY, Y_VAL_NPY, Y_TEST_NPY,
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pairs = [
        (X_train, X_TRAIN_NPY),
        (X_val,   X_VAL_NPY),
        (X_test,  X_TEST_NPY),
        (y_train, Y_TRAIN_NPY),
        (y_val,   Y_VAL_NPY),
        (y_test,  Y_TEST_NPY),
    ]
    for array, filename in pairs:
        save_numpy_array(array, output_dir / filename)

    logger.info(
        "Processed arrays saved to %s — "
        "X_train %s | X_val %s | X_test %s",
        output_dir,
        X_train.shape,
        X_val.shape,
        X_test.shape,
    )


def load_processed_arrays(
    input_dir: Union[str, Path],
) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray,
    np.ndarray, np.ndarray, np.ndarray,
]:
    """
    Load all six processed split arrays from *input_dir*.

    Returns
    -------
    tuple
        ``(X_train, X_val, X_test, y_train, y_val, y_test)``

    Raises
    ------
    FileNotFoundError
        If any of the six expected files are missing.
    """
    from src.utils.constants import (
        X_TRAIN_NPY, X_VAL_NPY, X_TEST_NPY,
        Y_TRAIN_NPY, Y_VAL_NPY, Y_TEST_NPY,
    )

    input_dir = Path(input_dir)

    X_train = load_numpy_array(input_dir / X_TRAIN_NPY)
    X_val   = load_numpy_array(input_dir / X_VAL_NPY)
    X_test  = load_numpy_array(input_dir / X_TEST_NPY)
    y_train = load_numpy_array(input_dir / Y_TRAIN_NPY)
    y_val   = load_numpy_array(input_dir / Y_VAL_NPY)
    y_test  = load_numpy_array(input_dir / Y_TEST_NPY)

    logger.info(
        "Processed arrays loaded — "
        "X_train %s | X_val %s | X_test %s",
        X_train.shape,
        X_val.shape,
        X_test.shape,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def load_split_data(
    input_dir: Union[str, Path],
) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray,
    np.ndarray, np.ndarray, np.ndarray,
    Any, Any,
]:
    """
    Load all six split arrays plus scaler and label encoder.

    Alias kept for backward compatibility with run_pipeline.py.

    Returns
    -------
    tuple
        ``(X_train, X_val, X_test, y_train, y_val, y_test,
           scaler, label_encoder)``
    """
    X_train, X_val, X_test, y_train, y_val, y_test = (
        load_processed_arrays(input_dir)
    )
    scaler = load_object(Path(input_dir) / "scaler.pkl")
    label_enc = load_object(Path(input_dir) / "label_encoder.pkl")
    return X_train, X_val, X_test, y_train, y_val, y_test, scaler, label_enc


def load_split_data_train(
    input_dir: Union[str, Path],
) -> Tuple[
    np.ndarray, np.ndarray,
    np.ndarray, np.ndarray,
    Any, Any,
]:
    """
    Load only train + val arrays (skip test) to reduce peak RAM.

    Use this during training/tuning/baselines.  Test data is
    loaded separately by ``load_split_data_test()`` when needed.

    Returns
    -------
    tuple
        ``(X_train, X_val, y_train, y_val, scaler, label_encoder)``
    """
    from src.utils.constants import (
        X_TRAIN_NPY, X_VAL_NPY,
        Y_TRAIN_NPY, Y_VAL_NPY,
    )

    input_dir = Path(input_dir)

    X_train = load_numpy_array(input_dir / X_TRAIN_NPY)
    X_val   = load_numpy_array(input_dir / X_VAL_NPY)
    y_train = load_numpy_array(input_dir / Y_TRAIN_NPY)
    y_val   = load_numpy_array(input_dir / Y_VAL_NPY)
    scaler  = load_object(input_dir / "scaler.pkl")
    label_enc = load_object(input_dir / "label_encoder.pkl")

    logger.info(
        "Train split loaded — X_train %s | X_val %s (test deferred)",
        X_train.shape, X_val.shape,
    )
    return X_train, X_val, y_train, y_val, scaler, label_enc


def load_split_data_test(
    input_dir: Union[str, Path],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load only test arrays — called on-demand during evaluation.

    Returns
    -------
    tuple
        ``(X_test, y_test)``
    """
    from src.utils.constants import X_TEST_NPY, Y_TEST_NPY

    input_dir = Path(input_dir)
    X_test  = load_numpy_array(input_dir / X_TEST_NPY)
    y_test  = load_numpy_array(input_dir / Y_TEST_NPY)

    logger.info("Test split loaded — X_test %s", X_test.shape)
    return X_test, y_test


# JSON Metadata Serialization

def save_metadata(
    metadata: Dict[str, Any],
    path: Union[str, Path],
) -> None:
    """
    Save a metadata dictionary as a human-readable JSON file.

    Used to persist dataset statistics, preprocessing
    parameters, feature counts, and model configuration
    alongside the saved model artifacts so that the inference
    pipeline can reconstruct the exact input specification
    used during training.

    Parameters
    ----------
    metadata : dict
        Arbitrary metadata dictionary.  NumPy scalars and
        Path objects are automatically converted.
    path : str or Path
        Destination ``.json`` file path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=4, default=_json_serialiser)
    logger.info("Metadata saved: %s", path)


def load_metadata(path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load a metadata JSON file.

    Parameters
    ----------
    path : str or Path
        Source ``.json`` file path.

    Returns
    -------
    dict

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Metadata file not found: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        metadata = json.load(fh)
    logger.info("Metadata loaded: %s", path)
    return metadata


# Feature Names Serialization

def save_feature_names(
    feature_names: List[str],
    path: Union[str, Path],
) -> None:
    """
    Save the ordered list of feature names as a pickle file.

    The feature names list is critical for the inference
    pipeline — it ensures that new input data is reindexed
    to match the exact column ordering used during training.

    Parameters
    ----------
    feature_names : list of str
        Ordered list of feature column names after all
        preprocessing transformations have been applied.
    path : str or Path
        Destination ``.pkl`` file path.
    """
    save_object(feature_names, path, use_joblib=True)
    logger.info(
        "Feature names saved (%d features): %s",
        len(feature_names),
        path,
    )


def load_feature_names(path: Union[str, Path]) -> List[str]:
    """
    Load the ordered list of feature names from a pickle file.

    Parameters
    ----------
    path : str or Path
        Source ``.pkl`` file path.

    Returns
    -------
    list of str
    """
    names = load_object(path, use_joblib=True)
    logger.info("Feature names loaded (%d features): %s", len(names), path)
    return names


# Bundled Artifact Save / Load
# (Convenience wrappers for the full artifact set)

def save_preprocessing_artifacts(
    scaler: Any,
    label_encoder: Any,
    feature_names: List[str],
    metadata: Dict[str, Any],
    output_dir: Union[str, Path],
) -> None:
    """
    Save the complete set of preprocessing artifacts to
    *output_dir* in a single call.

    Writes:
    - ``scaler.pkl``         — fitted MinMaxScaler
    - ``label_encoder.pkl``  — fitted LabelEncoder
    - ``feature_names.pkl``  — ordered feature name list
    - ``metadata.json``      — dataset / preprocessing stats

    Parameters
    ----------
    scaler : sklearn.preprocessing.MinMaxScaler
        Fitted scaler (trained on training set only).
    label_encoder : sklearn.preprocessing.LabelEncoder
        Fitted label encoder.
    feature_names : list of str
        Ordered feature name list after preprocessing.
    metadata : dict
        Dataset statistics and configuration snapshot.
    output_dir : str or Path
        Destination directory (typically ``data/processed/``
        or ``models/final/``).
    """
    from src.utils.constants import (
        SCALER_PKL, LABEL_ENCODER_PKL,
        FEATURE_NAMES_PKL, METADATA_JSON,
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    save_object(scaler, output_dir / SCALER_PKL)
    save_object(label_encoder, output_dir / LABEL_ENCODER_PKL)
    save_feature_names(feature_names, output_dir / FEATURE_NAMES_PKL)
    save_metadata(metadata, output_dir / METADATA_JSON)

    logger.info(
        "All preprocessing artifacts saved to: %s", output_dir
    )


def load_preprocessing_artifacts(
    input_dir: Union[str, Path],
) -> Tuple[Any, Any, List[str], Dict[str, Any]]:
    """
    Load the complete set of preprocessing artifacts from
    *input_dir* in a single call.

    Returns
    -------
    tuple
        ``(scaler, label_encoder, feature_names, metadata)``

    Raises
    ------
    FileNotFoundError
        If any artifact file is missing in *input_dir*.
    """
    from src.utils.constants import (
        SCALER_PKL, LABEL_ENCODER_PKL,
        FEATURE_NAMES_PKL, METADATA_JSON,
    )

    input_dir = Path(input_dir)

    scaler        = load_object(input_dir / SCALER_PKL)
    label_encoder = load_object(input_dir / LABEL_ENCODER_PKL)
    feature_names = load_feature_names(input_dir / FEATURE_NAMES_PKL)
    metadata      = load_metadata(input_dir / METADATA_JSON)

    logger.info(
        "All preprocessing artifacts loaded from: %s", input_dir
    )
    return scaler, label_encoder, feature_names, metadata


# Baseline Model Serialization

def save_baseline_model(
    model: Any,
    model_name: str,
    output_dir: Union[str, Path],
) -> None:
    """
    Save a trained scikit-learn baseline model.

    Parameters
    ----------
    model : sklearn estimator
        Fitted baseline model (RandomForest, SVM, etc.).
    model_name : str
        Identifier used to name the file, e.g. ``"random_forest"``.
    output_dir : str or Path
        Destination directory (typically ``models/baselines/``).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{model_name}.pkl"
    save_object(model, path)
    logger.info("Baseline model '%s' saved: %s", model_name, path)


def load_baseline_model(
    model_name: str,
    input_dir: Union[str, Path],
) -> Any:
    """
    Load a saved scikit-learn baseline model by name.

    Parameters
    ----------
    model_name : str
        Identifier used when the model was saved,
        e.g. ``"random_forest"``.
    input_dir : str or Path
        Directory containing the ``.pkl`` file
        (typically ``models/baselines/``).

    Returns
    -------
    sklearn estimator
        The loaded baseline model.
    """
    path = Path(input_dir) / f"{model_name}.pkl"
    return load_object(path)


def save_baseline_results(
    results: Dict[str, Any],
    output_dir: Union[str, Path],
) -> None:
    """
    Save baseline model evaluation results as JSON.

    Parameters
    ----------
    results : dict
        ``{model_name: {metric: value}}`` nested dictionary.
    output_dir : str or Path
        Destination directory (typically ``models/baselines/``).
    """
    from src.utils.constants import BASELINE_RESULTS_JSON

    path = Path(output_dir) / BASELINE_RESULTS_JSON
    save_metadata(results, path)
    logger.info("Baseline results saved: %s", path)