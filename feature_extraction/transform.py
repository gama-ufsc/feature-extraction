from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
from feature_extraction.config_models import TransformConfig


class Transform(BaseEstimator, TransformerMixin):
    """
    A transformer that applies signal representation and metrics to input data.

    Parameters
    ----------
    signal_representation : Callable
        A function that applies signal representation to input data.
    metrics : List[Callable]
        A list of functions that apply metrics to the signal representation.
    **kwargs
        Additional keyword arguments to pass to the transformer.

    Attributes
    ----------
    signal_representation : Callable
        A function that applies signal representation to input data.
    metrics : List[Callable]
        A list of functions that apply metrics to the signal representation.
    kwargs : dict
        Additional keyword arguments to pass to the transformer.

    Methods
    -------
    transform(X, y=None)
        Applies signal representation and metrics to input data.
    fit(X, y=None)
        Fits the transformer to input data.
    get_filter_name()
        Returns the name of the filter applied to the signal representation.
    """

    def __init__(self, config: TransformConfig):
        """
        Initializes the Transform object.

        Parameters
        ----------
        config: TransformConfig
            The configuration for the transformer using Pydantic.
        """
        self.config = config
        self.signal_representation = config.signal_representation
        self.metrics = config.metrics

    def transform(self, X, y=None):
        """
        Applies signal representation and metrics to input data.

        Parameters
        ----------
        X : array-like or sparse matrix of shape (n_samples, n_features)
            The input data.
        y : array-like of shape (n_samples,), default=None
            The target data.

        Returns
        -------
        X : pandas DataFrame
            The transformed data.
        """
        representation_params = self.signal_representation.params
        representation_name = self.signal_representation.name
        filter_name = self.get_filter_name()
        
        transformation = self.signal_representation.func(X, **representation_params)
        features_df = pd.DataFrame()

        for metric in self.metrics:
            metric_params = metric.params
            metric_name = metric.name

            metric_results = self._metric_calculation(
                X, metric.func, transformation, metric_params
            )

            harmonics = metric_params.get("harmonics", [])

            if len(metric_results.columns) == 1 & (harmonics == []):
                metric_results = metric_results.rename(
                    columns={0: f"{representation_name}/{metric_name}/{filter_name}"}
            ) if filter_name is not None else metric_results.rename(
                    columns={0: f"{representation_name}/{metric_name}"}
            )
                
            elif len(metric_results.columns) == 1 & (harmonics != []):
        
                if (filter_name == 'global') | (filter_name == 'custom'):
                            filter_name = None

                metric_results = metric_results.rename(
                    columns={0: f"{representation_name}/{harmonics[0]}x-{metric_name}/{filter_name}"}
                ) if filter_name is not None else metric_results.rename(
                    columns={0: f"{representation_name}/{harmonics[0]}x-{metric_name}"})
                
            elif len(metric_results.columns) > 1:
                # TODO: Define a logic for naming sidebands

                feature_identifier = metric_params.get("feature_identifier", "")
                harmonics = metric_params.get("harmonics", [])

                if harmonics != []:  # if harmonics != [] and len(harmonics) > 1:
                    # LOGIC FOR NAMING SIDE BANDS
                    if type(feature_identifier) == dict:
                        fundamental_frequency_nickname = feature_identifier.get(
                            "fundamental_frequency_nickname", ""
                        )
                        sideband_frequency_nickname = feature_identifier.get(
                            "sideband_frequency_nickname", ""
                        )
                        sideband_harmonics = metric_params.get("sideband_harmonics", [])
                        column_names = [
                            f"{representation_name}/{harmonics[i]}*{fundamental_frequency_nickname}{sideband_harmonics[j]}{sideband_frequency_nickname}"
                            if sideband_harmonics[j]
                            < 0  # para considerar o sinal de sidebandfrequency
                            else f"{representation_name}/{harmonics[i]}*{fundamental_frequency_nickname}+{sideband_harmonics[j]}{sideband_frequency_nickname}"
                            for i in range(len(harmonics))
                            for j in range(len(sideband_harmonics))
                        ]

                        metric_results = metric_results.rename(
                            columns={i: column_names[i] for i in metric_results.columns}
                        )

                    elif len(harmonics) > 1:

                        if (filter_name == 'global') | (filter_name == 'custom'):
                            filter_name = None
 
                        metric_results = metric_results.rename(
                        columns={
                            i: f"{representation_name}/{float(harmonics[i])}x-{metric_name}/{filter_name}"
                            for i in metric_results.columns
                        }
                        ) if filter_name is not None else metric_results.rename(
                        columns={
                            i: f"{representation_name}/{float(harmonics[i])}x-{metric_name}"
                            for i in metric_results.columns
                        })

                else:

                    if isinstance(metric_results.columns,pd.RangeIndex):
                        metric_results.columns += 1

                    metric_results = metric_results.rename(
                        columns={
                            i: f"{representation_name}/{i}.0x-{metric_name}/{filter_name}"
                            for i in metric_results.columns
                        }
                    ) if filter_name is not None else metric_results.rename(
                        columns={
                            i: f"{representation_name}/{i}.0x-{metric_name}"
                            for i in metric_results.columns
                        }
                    )

            features_df = pd.concat([features_df, metric_results], axis=1)

        # No metrics were applied
        if len(features_df) == 0:
            features_df = transformation

        X = pd.concat([X, features_df], axis=1)

        return X

    def fit(self, X, y=None):
        """
        Fits the transformer to input data.

        Parameters
        ----------
        X : array-like or sparse matrix of shape (n_samples, n_features)
            The input data.
        y : array-like of shape (n_samples,), default=None
            The target data.

        Returns
        -------
        self : object
            Returns self.
        """
        return self

    def get_filter_name(self) -> str:
        """
        Returns the name of the filter applied to the signal representation.

        Returns
        -------
        filter_name : str
            The name of the filter applied to the signal representation.
        """
        representation_params = self.signal_representation.params

        if representation_params.get("cutoff") is None:
            return "global"
        elif representation_params.get("cutoff") == "kurtogram":
            return "kurtogram"
        elif isinstance(representation_params.get("cutoff"), list):
            cutoff = representation_params.get("cutoff")
            return f"{cutoff[0]}-{cutoff[1]}"

    def _metric_calculation(self, X, metric_func, transformation, metric_params):
        # Number of obrigatory parameters
        num_obrig_params = (
            metric_func.__code__.co_argcount - len(metric_func.__defaults__)
            if metric_func.__defaults__
            else metric_func.__code__.co_argcount
        )

        if num_obrig_params == 1:
            metric_results = transformation.apply(
                lambda x: pd.Series(
                    metric_func(
                        x.iloc[0],
                        **metric_params,
                    )
                ),
                axis=1,
            )

        elif num_obrig_params > 1:
            X_columns = metric_params["X_columns"]
            X = X[X_columns].copy().reset_index(drop=True)
            transformation = transformation.reset_index(drop=True)
            transformation = pd.concat([transformation, X], axis=1)

            try:
                metric_results = transformation.apply(
                    lambda x: pd.Series(
                        metric_func(
                            *[x.iloc[n] for n in range(len(transformation.columns))],
                            **metric_params,
                        )
                    ),
                    axis=1,
                )

            except Exception as e:
                print(e)
                print(f"Error in {metric_func.__name__} with {transformation.columns}")
                metric_results = pd.DataFrame()

        return metric_results
