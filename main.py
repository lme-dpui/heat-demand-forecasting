"""Entry point for the wavelet-scalogram heat-demand forecasting pipeline.

Hydra loads ``conf/config.yaml`` into a :class:`WaveletConfig`; ``main`` then
builds the datasets, runs training/forecasting, and writes metrics and forecast
plots to the Hydra run's output directory.
"""
from omegaconf import OmegaConf
import hydra
from hydra.core.config_store import ConfigStore
import logging
from config import WaveletConfig
import src.utils as utils
from src.dataset import HeatDataset, split_data_train_validation_test, HistoricalDataset
import torch
from torch.utils.data import DataLoader
from src.model.model import initiate_training
import numpy as np

def update_log_file(config_data):
    """Log a summary of the key run settings for this experiment."""
    log.info(f"Dataset: {config_data.paths.dma}")
    log.info(f"Historical Window: {config_data.time.historical_values}")
    log.info(f"Prediction Window: {config_data.time.prediction_horizon}")
    log.info(f"Architecture: {config_data.setup.architecture.network}")
    log.info(f"Method: {config_data.setup.encoding_schema.encoding_type}")
    if config_data.setup.encoding_schema.encoding_type == "wavelet":
        log.info(f"Wavelet Function: {config_data.setup.encoding_schema.wavelet_function}")
    log.info(f"Length of test data: {config_data.time.data_length_in_weeks_test} weeks | {config_data.time.data_length_in_weeks_test*7} days | {config_data.time.data_length_in_weeks_test*7*24} hours ")
    log.info(f"Feature subset: {config_data.variables.features}")



def generate_data_for_model(config_data):
    """Build the train/validation/test datasets and their DataLoaders.

    Loads and encodes the raw demand + weather data, splits it into
    train/validation/test, wraps each split in a :class:`HistoricalDataset`,
    and returns the loaders together with the source dataset and the
    test-window timestamps needed downstream.
    """
    dataset = HeatDataset(config_data)

    input_timeseries, y, df_demand_en, time_test_array = dataset.generate_input_data_for_network()
    log.info(f"Input path: {dataset.timeseries_path}")
    log.info(f"Feature channels ({dataset.number_of_features}): {list(dataset.feature_variables)}")

    training_data, validation_data, test_data = split_data_train_validation_test(config_data,
                                                                                 input_timeseries,
                                                                                 y)

    log.info(f"Shape of Training Data | Timeseries: {training_data['time'].shape}, Truth: {training_data['truth'].shape}")
    log.info(f"Shape of Validation Data | Timeseries: {validation_data['time'].shape}, Truth: {validation_data['truth'].shape}")
    log.info(f"Shape of Test Data | Timeseries: {test_data['time'].shape}, Truth: {test_data['truth'].shape}")

    train_dataset = HistoricalDataset(training_data, config_data, 'train')
    train_loader = DataLoader(train_dataset, batch_size=config_data.model_params.batch_size, shuffle=True, pin_memory=True)

    validation_dataset = HistoricalDataset(validation_data, config_data, 'validation', train_dataset.min_value, train_dataset.max_value)
    validation_loader = DataLoader(validation_dataset, batch_size=config_data.model_params.batch_size, shuffle=False, pin_memory=True)

    test_dataset = HistoricalDataset(test_data, config_data, 'test', train_dataset.min_value, train_dataset.max_value)
    test_loader = DataLoader(test_dataset, batch_size=config_data.model_params.batch_size, shuffle=False, pin_memory=True)

    return dataset, df_demand_en, time_test_array, train_loader, validation_loader, test_loader


OmegaConf.register_new_resolver('sanitize_override_dirname', lambda x: x.replace('/', '_'))

cs = ConfigStore.instance()
cs.store(name="wavelet_config", node=WaveletConfig)

log = logging.getLogger(__name__)


@hydra.main(config_path="conf", config_name="config.yaml", version_base=None)
def main(config_data: WaveletConfig):
    torch.manual_seed(99)  # seed first, before any data prep, so the run is reproducible
    np.random.seed(42)

    # Create the per-run output subdirectories under this Hydra run's directory.
    utils.create_folders_for_results()
    print(OmegaConf.to_yaml(config_data))
    update_log_file(config_data)

    print(f"Is GPU available: {torch.cuda.is_available()}")
    print(f"Num GPUs Available: {torch.cuda.device_count()}")

    # Build the datasets/loaders, then train and forecast the held-out test window.
    dataset, df_demand_en, test_timeframe, train_loader, validation_loader, test_loader = generate_data_for_model(config_data)

    forecast_trainer, result_array, validation_truth, validation_prediction = initiate_training(config_data, train_loader, validation_loader, test_loader, dataset.consumption_scaler, test_timeframe)

    # Log the training summary and plot the loss curves.
    log.info(f"Best Epoch: {forecast_trainer.best_epoch}")
    log.info(f"Training Time: {forecast_trainer.total_training_time}")
    utils.plot_loss_curves(forecast_trainer.training_loss.values, forecast_trainer.validation_loss.values)

    # Score the best model on the validation and test splits and write the metric artifacts.
    validation_metric_df = utils.process_validation_metrics(validation_truth, validation_prediction)
    metric_df = utils.process_error_metrics(result_array)

    log.info(f"Validation | RMSE: {validation_metric_df.loc['rmse', 0].item()} | "
             f"MAPE: {validation_metric_df.loc['mape', 0].item()} | "
             f"MAE: {validation_metric_df.loc['mae', 0].item()} | "
             f"MSE: {validation_metric_df.loc['mse', 0].item()}")
    log.info(f"RMSE of the single timeseries: {metric_df.loc['rmse', 0].item()}")
    log.info(f"MAPE of the single timeseries: {metric_df.loc['mape', 0].item()}")
    log.info(f"MAE of the single timeseries: {metric_df.loc['mae', 0].item()}")
    log.info(f"MSE of the single timeseries: {metric_df.loc['mse', 0].item()}")

    utils.plot_forecast_with_ground_truth(config_data, result_array, bool(config_data.variables.plot_forecast_individually))
    utils.plot_forecast_timeseries(config_data, result_array)


if __name__ == '__main__':
    main()
