<div align="center"> 

# FeatureExtraction 

</div>

## Description

Library for implementing different representations and metrics for vibration signals. The goal is to facilitate feature extraction from signals and integrate with the [scikit-learn](https://github.com/scikit-learn/scikit-learn) library using the [Pipeline](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html) function.

A signal representation is a transformation that accepts a set of vectors (a pd.DataFrame where each row contains a signal vector) and returns a set of vectors with a new representation, such as the FFT of the signal. A metric, in turn, accepts a set of vectors (a pd.DataFrame where each row contains a signal vector) and returns scalars, which are referred to as features of the representation. For example, the RMS value of the filtered signal or its peak-to-peak value.

## Installation

Installation is done by referencing this GitHub repository (the package has not been added to PyPI). For this, you must be logged into your Git account, since the repository is private.

The library dependencies can be found in the `pyproject.toml` file. To install the library, you can use either `pip` or `poetry`.

### Pip
```
pip install git+https://github.com/gama-ufsc/feature-extraction
```

### Poetry

With `poetry`, the same logic applies as when using `pip`:
```
poetry add git+https://github.com/gama-ufsc/feature-extraction
```

## Usage

The library provides a main class called `Transform`, which inherits from the `TransformerMixin` class from `scikit-learn`. This class is responsible for receiving a signal representation and a list of desired metrics, and returning a `pd.DataFrame` with the extracted features using the `fit_transform` method. To initialize this class, it is necessary to use the `TransformConfig` object, which takes the signal representation as a `SignalRepresentationConfig` object and a list of metrics, each one of type `MetricConfig`.

```python
from feature_extraction import Transform, TransformConfig,
SignalRepresentationConfig, MetricConfig
from feature_extraction.metrics.statistics import rms, pk_pk
from feature_extraction.signal_representations import fft_transform, bandpass_filter_transform

# Signal representation configuration
signal_representation_config = SignalRepresentationConfig(
    func=bandpass_filter_transform,
    params={
        "signal_column": "signal",
        "sampling_rate_column": "sampling_rate",
        "cutoff": [100, 1000] # Hz
    },
    name="bandpass_filter"
)

# Metric configuration
metric_rms = MetricConfig(func=rms, name="rms")
metric_pkpk = MetricConfig(func=pk_pk, name="pkpk")

# Transform configuration
transform_config = TransformConfig(
    signal_representation=signal_representation_config,
    metrics=[metric_rms, metric_pkpk]
)

# Instantiating the transform class
transform = Transform(transform_config)

# Applying the transformation
features = transform.fit_transform(data)
```

For more examples integrating with `scikit-learn`, see the notebook `examples/transform_usage.ipynb`.