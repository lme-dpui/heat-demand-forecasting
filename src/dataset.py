"""Dataset assembly for the wavelet-scalogram heat-demand pipeline.

Turns the featured hourly dataframe (built by :mod:`src.features`) into the
tensors the CNN consumes. Three pieces:

* :class:`HeatDataset` — loads and features the raw CSVs, slices the configured
  date range, and windows the hourly series into per-day samples of shape
  ``[features, samples, lookback]`` plus the aligned next-day targets.
* :func:`split_data_train_validation_test` — carves those samples into
  train/validation/test sets along the sample axis.
* :class:`HistoricalDataset` — a ``torch`` ``Dataset`` that turns each windowed
  sample into a multi-channel CWT scalogram image, min-max normalised using
  train-split statistics.
"""
import numpy as np
import pandas as pd
import src.image_representation as image_representation
import src.features as features
import src.feature_names as feature_names
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset
from definitions import DATA_DIR
from pathlib import Path


class HeatDataset:
    """Load, feature-engineer and window the demand series into model inputs.

    Given a run config, resolves the demand/weather CSV paths, delegates feature
    construction to :mod:`src.features`, then slices the configured date range and
    windows the hourly data into per-day samples. The consumption scaler fitted
    here on the training split is reused downstream to invert forecasts.
    """

    def __init__(self, config_data):
        self.config_data = config_data  # config file

        self.data_key = config_data.paths.dma.lower()

        # Demand CSVs follow the flat published dataset layout (Zenodo DOI
        # 10.5281/zenodo.17398331): `<dma>_norm_aggregated.csv` (average
        # consumption per active meter) and `<dma>_aggregated.csv` (total
        # consumption summed across meters), both under DATA_DIR.
        if self.config_data.paths.summed_averaged:
            self.timeseries_path = Path(DATA_DIR, self.data_key + "_norm_aggregated.csv")
        else:
            self.timeseries_path = Path(DATA_DIR, self.data_key + "_aggregated.csv")

        self.look_back = config_data.time.historical_values  # look back window
        self.horizon = config_data.time.prediction_horizon  # prediction horizon of output

        self.start_time_of_data = config_data.time.start_time_train  # starting time of the data
        self.end_time_of_data = config_data.time.end_time_test  # ending time of the data

        self.encoding_type = config_data.setup.encoding_schema.encoding_type  # options between wavelet, superlet-freq, superlet-scale
        self.wavelet_function = config_data.setup.encoding_schema.wavelet_function  # optional between different wavelet function -> mexh, morl, gaus, etc.
        self.num_frequencies = config_data.setup.encoding_schema.num_frequencies  # determines the array size - defines the range of scales/frequencies

        self.feature_variables = feature_names.expand_feature_spec(config_data.variables.features)  # subset spec -> ordered column names
        self.number_of_features = len(self.feature_variables)  # number of features

        self.consumption_scaler = preprocessing.MinMaxScaler()

    def scale_supporting_features(self, df_demand_en):
        """Min-max scale every feature column into ``self.scaled_features``.

        Consumption-derived columns (name contains ``'cons'``) share the
        consumption scaler already fitted on the training split, so they stay on
        the same scale as the target; every other feature gets its own scaler fit
        on the full column.

        :param df_demand_en: dataframe of the selected feature columns
        """

        scalers = {}
        self.scaled_features = df_demand_en.copy()
        for feat in df_demand_en.columns:
            if 'cons' in feat:

                self.scaled_features.loc[:, feat] = self.consumption_scaler.transform(np.array(df_demand_en[feat]).reshape([-1, 1])).ravel()

                scalers[feat] = self.consumption_scaler
            else:
                f_scaler = preprocessing.MinMaxScaler()
                self.scaled_features.loc[:, feat] = f_scaler.fit_transform(np.array(df_demand_en[feat]).reshape([-1, 1])).ravel()
                scalers[feat] = f_scaler

    def populate_input_vectors(self, df_demand_en: pd.DataFrame, input_features, y) -> (np.ndarray, np.ndarray):
        """
        Fill in the initialised array with feature data and ground truth data appropriately
        :param df_demand_en: dataframe with feature values
        :param input_features: initialised multidimensional array - [number of features, number of samples (or) days, lookback window size]
        :param y: initialised ground truth data - [number of samples (or) days, prediction horizon size]
        :return: populated input array and ground truth array
        """

        train_data_subset = self.consumption_values[:-self.config_data.time.data_length_in_weeks_test * 24 * 7]  # take the training data

        self.consumption_scaler.fit(train_data_subset)  # update the scaler with training data
        cons = self.consumption_scaler.transform(self.consumption_values)  # update the consumption values - the whole dataset

        self.scale_supporting_features(df_demand_en)

        self.test_time_array = np.empty(shape=[input_features.shape[1], self.config_data.time.prediction_horizon])

        for jj in range(0, input_features.shape[1], 1):  # iterate through all samples

            input_start = jj * self.config_data.time.prediction_horizon
            input_end = jj * self.config_data.time.prediction_horizon + self.config_data.time.historical_values
            output_start = jj * self.config_data.time.prediction_horizon + self.config_data.time.historical_values
            output_end = jj * self.config_data.time.prediction_horizon + self.config_data.time.historical_values + self.config_data.time.prediction_horizon

            input_features[:, jj, :] = self.scaled_features[input_start:input_end].T
            y[jj] = cons[output_start:output_end].reshape([-1])
            # Store target timestamps as int64 nanoseconds-since-epoch. Forcing ns keeps
            # the values correct whatever resolution the index carries (pandas >=2 may use
            # us/s), since downstream code reads them back with a unitless pd.to_datetime
            # that assumes ns.
            self.test_time_array[jj] = df_demand_en.index[output_start:output_end].values.astype('datetime64[ns]').astype(np.int64)

        return input_features, y

    def initialise_input_vectors(self, df_demand_en: pd.DataFrame) -> (np.ndarray, np.ndarray):
        """
        initialise ground truth and input arrays with zeros
        :param df_demand_en: base dataframe
        :return: initialised input features and ground truth array
        """
        dicretized = int(len(df_demand_en) / self.config_data.time.prediction_horizon)  # get the number of samples based on lookback window size
        number_of_possible_samples = int((dicretized * self.config_data.time.prediction_horizon - self.config_data.time.historical_values) / self.config_data.time.prediction_horizon) \
                                     - int(self.config_data.time.prediction_horizon / self.config_data.time.prediction_horizon) + 1

        y = np.zeros((number_of_possible_samples, self.config_data.time.prediction_horizon))  # initialise ground truth -> consumption of next day
        input_features = np.zeros([self.number_of_features, number_of_possible_samples, self.config_data.time.historical_values])  # initialise input array

        return input_features, y

    def create_inputs_as_vectors(self, df_demand_en: pd.DataFrame) -> (np.ndarray, np.ndarray):
        """
        Initialise array and populate the array according to number of samples and features
        :param df_demand_en: input dataframe with specified features
        :return: populate input arrays and ground truth array
        """

        input_features, y = self.initialise_input_vectors(df_demand_en)
        input_features, y = self.populate_input_vectors(df_demand_en, input_features, y)

        return input_features, y

    def generate_input_data_for_network(self):
        """
        Load, feature-engineer and window the data into the model's input arrays.

        Reads the demand CSV, merges weather and calendar features, encodes
        holiday/cyclical information, restricts to the configured date range
        (aligned to whole days and whole weeks), then windows into samples.

        :return: tuple of (input windows ``[features, samples, lookback]``,
            targets ``[samples, horizon]``, the sliced feature dataframe,
            and the per-sample target timestamps)
        """

        df_demand_en = features.read_demand_data(self.timeseries_path)  # read the consumption data
        df_demand_en = features.weather_day_merge(df_demand_en, self.config_data.paths.weather_data_path)  # merge zonal heat demand with weather and calendar info
        # `use_previous_holiday` (the paper's h_lagged strategy) is a per-feature-set
        # switch living in the selected set's `calendar` block, so it can differ per
        # DMA (see conf/features/). Absent -> off.
        calendar_spec = self.config_data.variables.features.get("calendar") or {}
        use_previous_holiday = bool(calendar_spec.get("use_previous_holiday", False))
        df_demand_en = features.encode_day_of_week_and_holidays(df_demand_en, use_previous_holiday)  # encode day/holiday info

        mask = (df_demand_en.index >= self.config_data.time.start_time_train) & \
               (df_demand_en.index < self.config_data.time.end_time_test)  # create a mask for certain defined date range
        df_demand_en = df_demand_en.loc[mask]  # filter out a subpart of the dataframe to be used in training and testing
        df_demand_en = df_demand_en[(df_demand_en.index >= df_demand_en[df_demand_en.index.hour == 0].index[0])]
        df_demand_en = df_demand_en[(df_demand_en.index <= df_demand_en[df_demand_en.index.hour == 23].index[-1])]
        df_demand_en = df_demand_en[(df_demand_en.index >= df_demand_en[df_demand_en.index.dayofweek == 0].index[0])]
        df_demand_en = df_demand_en[(df_demand_en.index <= df_demand_en[df_demand_en.index.dayofweek == 6].index[-1])]

        self.consumption_values = np.array(df_demand_en['cons']).reshape([-1, 1])  # just the hourly consumption values
        df_demand_en = df_demand_en[self.feature_variables]  # slice out only the required features from the total feature set

        input_features, y = self.create_inputs_as_vectors(df_demand_en)  # initialise and populate arrays that serve as a bridge to create the image-like representations

        return input_features, y, df_demand_en, self.test_time_array


def split_data_train_validation_test(config_data, input_timeseries, y):
    """
    Split the windowed samples into train/validation/test sets along the sample axis.

    :param config_data: run configuration (drives horizon, test length, split ratios)
    :param input_timeseries: input windows, shape [features, samples, lookback]
    :param y: target sequences, shape [samples, prediction_horizon]
    :return: train, validation and test dicts, each with "time" and "truth" arrays
    """

    input_timeseries = np.transpose(input_timeseries, (1, 2, 0))

    # Number of most-recent samples held out for the test set. For the day-ahead
    # horizon this is a fixed number of weeks; for the week-ahead horizon it is a
    # fixed fraction of the data.
    if config_data.time.prediction_horizon == 24:
        num_test_samples = config_data.time.data_length_in_weeks_test * 7
    elif config_data.time.prediction_horizon == 168:
        num_test_samples = int(len(input_timeseries) * 0.2)
    else:
        raise ValueError(f"Unsupported prediction_horizon: {config_data.time.prediction_horizon} (expected 24 or 168)")

    X_test_time, y_test = input_timeseries[-num_test_samples:], y[-num_test_samples:]  # most recent samples -> test

    X_train_time = input_timeseries[:-num_test_samples]  # everything before the test window -> train + validation
    y_train = y[:-num_test_samples]

    train_data_portion = int(len(X_train_time) * config_data.time.percentage_of_train_data)  # keep only the tail fraction of the train pool (tuned hyperparameter)
    X_train_time = X_train_time[-train_data_portion:]
    y_train = y_train[-train_data_portion:]

    X_train_time, X_validation_time, y_train, y_validation = train_test_split(X_train_time,
                                                                              y_train,
                                                                              train_size=1 - config_data.time.validation_period,
                                                                              test_size=config_data.time.validation_period,
                                                                              shuffle=True, random_state=42)

    training_data = {"time": X_train_time, "truth": y_train}
    validation_data = {"time": X_validation_time, "truth": y_validation}
    test_data = {"time": X_test_time, "truth": y_test}

    return training_data, validation_data, test_data


class HistoricalDataset(Dataset):
    """Torch dataset yielding (scalogram image, target) pairs for one split.

    Each windowed sample is transformed into a multi-channel time-frequency image
    (one channel per feature) via a batched CWT, then min-max normalised per
    channel. The ``'train'`` split computes and stores the per-channel
    ``min_value``/``max_value``; the ``'validation'`` and ``'test'`` splits must
    be passed those same statistics so they are normalised on the training scale.

    :param train_data: dict with ``'time'`` (windows) and ``'truth'`` (targets)
    :param config: run configuration (encoding schema, lookback, ...)
    :param stage: one of ``'train'``, ``'validation'``, ``'test'``
    :param min_value/max_value: per-channel train-split stats (required off-train)
    """

    def __init__(self, train_data, config, stage='train', min_value=None, max_value=None):

        self.X_timeseries, self.y = train_data['time'], train_data['truth']
        self.config = config
        self._stage = stage


        if self._stage == 'train':
            self.min_value, self.max_value = np.zeros(shape=self.X_timeseries.shape[-1]), np.zeros(shape=self.X_timeseries.shape[-1])
        elif (self._stage == 'validation') or (self._stage == 'test'):
            self.min_value, self.max_value = min_value, max_value

        # desc: Create image features for training data
        # region
        image_features = np.zeros([self.X_timeseries.shape[0], config.setup.encoding_schema.num_frequencies, config.time.historical_values,
                                   self.X_timeseries.shape[-1]])  # initialise array to store the image-like features [number of day samples, img_x, img_y, number of features]
        # Transform every (day, feature) window in a single batched CWT along the
        # time axis instead of looping in Python; the leading scale axis is then
        # moved into position [days, num_freq, lookback, features].
        scalograms = image_representation.create_image_representation(self.X_timeseries,
                                                                      config.setup.encoding_schema,
                                                                      config.setup.encoding_schema.num_frequencies,
                                                                      method=config.setup.encoding_schema.encoding_type,
                                                                      axis=1)  # [num_freq, days, lookback, features]
        image_features[:] = np.moveaxis(scalograms, 0, 1)

        # endregion


        # desc: Normalise data based on train data statistics
        # region
        if self._stage == 'train':
            for i in range(image_features.shape[3]):  # iterate through image feature and feature scale them.
                self.min_value[i], self.max_value[i] = image_features[:, :, :, i].min(), image_features[:, :, :, i].max()
                image_features[:, :, :, i] = (image_features[:, :, :, i] - image_features[:, :, :, i].min()) / (
                        image_features[:, :, :, i].max() - image_features[:, :, :, i].min())
        elif self._stage == 'validation':
            for i in range(image_features.shape[3]):  # iterate through image feature and feature scale them.
                image_features[:, :, :, i] = (image_features[:, :, :, i] - self.min_value[i]) / (self.max_value[i] - self.min_value[i])
        elif self._stage == 'test':
            for i in range(image_features.shape[3]):  # iterate through image feature and feature scale them.
                image_features[:, :, :, i] = (image_features[:, :, :, i] - self.min_value[i]) / (self.max_value[i] - self.min_value[i])
        # endregion

        self.X_image = np.transpose(image_features, (0, 3, 1, 2))

        # desc: Convert arrays to Torch
        # region
        self.y = torch.tensor(self.y).float()
        self.X_timeseries = torch.tensor(self.X_timeseries).float()
        self.X_image = torch.tensor(self.X_image).float()
        # endregion

    def __len__(self):
        return self.X_timeseries.shape[0]

    def __getitem__(self, i):
        return self.X_image[i], self.y[i]


