from dataclasses import dataclass
from typing import Any


@dataclass
class Info:
    comments: str
    experiment_no: str


@dataclass
class Paths:
    dma: str
    summed_averaged: bool
    weather_data_path: str  # path to weather data of Bronderslev


@dataclass
class Variables:
    features: Any  # feature-subset spec (demand/weather/calendar blocks); see conf/features/
    plot_forecast_individually: bool
    plot_sample_size: int


@dataclass
class Parameters:
    epoch: int  # maximum number of epochs
    restore_epoch: int
    lr: float  # learning rate
    lr_scheduler_factor: float  # ReduceLROnPlateau decay factor applied on a validation-loss plateau
    batch_size: int  # batch size
    patience: int  # early stopping criteria
    loss_function: str


@dataclass
class Time:
    start_time_train: str  # starting time of the data
    end_time_test: str  # test period starting time
    percentage_of_train_data: float  # length of the train period
    data_length_in_weeks_test: int  # length of the test period
    historical_values: int  # lookback window
    prediction_horizon: int  # forecasting window
    validation_period: float  # validation split (%/100 -> between 0 to 1)


@dataclass
class Encoding:
    encoding_type: str
    wavelet_function: Any  # default mother wavelet basis function
    num_frequencies: int  # number of scales for 24 step ahead prediction


@dataclass
class Setup:
    architecture: Any
    encoding_schema: Encoding


@dataclass
class WaveletConfig:
    info: Info
    paths: Paths
    variables: Variables
    model_params: Parameters
    time: Time
    setup: Setup
