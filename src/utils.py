"""Post-processing helpers: error metrics, result serialisation, and plotting.

These functions run after training/forecasting (see ``main.py``) to score the
held-out test window, write the metric/prediction artifacts into the Hydra run
directory, and produce the accompanying figures.
"""
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
from statsmodels.tsa.seasonal import DecomposeResult, seasonal_decompose
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tqdm import tqdm

from definitions import (
    METRICS_DIRNAME,
    MODEL_DIRNAME,
    PREDICTIONS_DIRNAME,
    TEST_DATA_DIRNAME,
    TRUTH_PREDICTION_DIRNAME,
    run_output_dir,
)


def plot_seasonal_decomposition_with_comparison(result:DecomposeResult, result_temp:DecomposeResult, dates:pd.Series=None, title:str="Seasonal Decomposition with Comparison"):
    """Overlay two seasonal decompositions (truth vs. prediction) as a 4-row figure.

    Each row shows one decomposition component (observed, trend, seasonal,
    residual) with the ``result`` and ``result_temp`` series drawn on the same
    axes for comparison.

    :param result: decomposition of the ground-truth series.
    :param result_temp: decomposition of the predicted series.
    :param dates: x-axis values; falls back to a positional index when omitted.
    :param title: figure title.
    :return: the assembled Plotly figure.
    """
    x_values = dates if dates is not None else np.arange(len(result.observed))
    return (
        make_subplots(
            rows=4,
            cols=1,
            subplot_titles=["Observed", "Trend", "Seasonal", "Residuals"],
        )
            .add_trace(
            go.Scatter(x=x_values, y=result.observed, mode="lines", name='Truth'),
            row=1,
            col=1,
        )
            .add_trace(
            go.Scatter(x=x_values, y=result_temp.observed, mode="lines", name='Prediction'),
            row=1,
            col=1,
        )  # ------------------------------------------------------------------------------------------------------------------
            .add_trace(
            go.Scatter(x=x_values, y=result.trend, mode="lines", name='Truth'),
            row=2,
            col=1,
        )
            .add_trace(
            go.Scatter(x=x_values, y=result_temp.trend, mode="lines", name='Prediction'),
            row=2,
            col=1,
        )  # ------------------------------------------------------------------------------------------------------------------
            .add_trace(
            go.Scatter(x=x_values, y=result.seasonal, mode="lines", name='Truth'),
            row=3,
            col=1,
        )
            .add_trace(
            go.Scatter(x=x_values, y=result_temp.seasonal, mode="lines", name='Prediction'),
            row=3,
            col=1,
        )  # ------------------------------------------------------------------------------------------------------------------
            .add_trace(
            go.Scatter(x=x_values, y=result.resid, mode="lines", name='Truth'),
            row=4,
            col=1,
        )
            .add_trace(
            go.Scatter(x=x_values, y=result_temp.resid, mode="lines", name='Prediction'),
            row=4,
            col=1,
        )  # ------------------------------------------------------------------------------------------------------------------

            .update_layout(
            height=900, title=f'<b>{title}</b>', margin={'t': 100}, title_x=0.5, showlegend=False
        )
    )



def write_data_to_file(config_data, dataset, model_rf):
    """Dump the train/test splits, raw arrays, unrolled results, and error log to disk.

    A debugging/archival helper for persisting a run's inputs and outputs
    alongside its metrics.

    Expects ``dataset`` to expose ``_dataset_train`` / ``_dataset_test`` frames
    and ``model_rf`` to expose ``_training_data`` / ``_test_data`` arrays,
    ``_results_in_order`` (+ ``test_dateindex``), and an ``_error_log``. It is
    not wired into ``main.py``'s current pipeline, so callers must supply
    objects matching that interface.
    """
    save_base_path = run_output_dir() + str(config_data.info.experiment_no)
    dataset._dataset_train.to_csv(save_base_path + "/dataset_train.csv")
    dataset._dataset_test.to_csv(save_base_path + "/dataset_test.csv")

    np.save(save_base_path + "/training_data_x.npy", model_rf._training_data[0])
    np.save(save_base_path + "/training_data_y.npy", model_rf._training_data[1])

    np.save(save_base_path + "/test_data_x.npy", model_rf._test_data[0])
    np.save(save_base_path + "/test_data_y.npy", model_rf._test_data[1])

    model_rf._results_in_order.index = model_rf.test_dateindex
    model_rf._results_in_order.to_csv(save_base_path + "/results_unrolled.csv")

    with open(save_base_path + "/log.txt", 'w') as f:
        for line in model_rf._error_log:
            f.write(line + "\n")


def plot_forecast_with_ground_truth(config_data, result_array, plot_individual_samples: bool = True):
    """Plot forecast vs. ground truth over the test window.

    Seasonally decomposes both the truth and prediction series and writes an
    interactive comparison of their components. When ``plot_individual_samples``
    is set, also saves one truth-vs-prediction figure per sampled test sequence.

    :param config_data: config object (used for the individual-sample count).
    :param result_array: array of [datetime (ns), ground_truth, prediction] per sample.
    :param plot_individual_samples: also plot individual test sequences when True.
    """
    save_base_path = run_output_dir() + PREDICTIONS_DIRNAME

    together = pd.DataFrame(np.concatenate(result_array, axis=0), columns=['Datetime', 'Truth', 'Prediction'])
    together.index = pd.to_datetime(together['Datetime'], utc=True)

    truth_sd = seasonal_decompose(together['Truth'], model='additive')
    pred_sd = seasonal_decompose(together['Prediction'], model='additive')
    fig_sd = plot_seasonal_decomposition_with_comparison(truth_sd, pred_sd, dates=together.index.values)
    fig_sd.write_html(save_base_path + f"/SeasonallyDecomposed_Truth_Predictions.html")


    # Space the individually-plotted samples evenly across the test window.
    step = int(len(result_array)/config_data.variables.plot_sample_size)
    if plot_individual_samples:
        os.mkdir(save_base_path + "/Individual_TestData")

        for i in tqdm(range(0, len(result_array), step), desc='Plot Truth vs Prediction for each test sequence'):

            date_value = pd.to_datetime(result_array[i, :, 0], utc=True)
            truth = result_array[i, :, 1]
            predictions = result_array[i, :, 2]

            truth = pd.DataFrame(truth, index=date_value, columns=["truth"])
            predictions = pd.DataFrame(predictions, index=date_value, columns=["predictions"])


            fig, ax = plt.subplots()
            plt.rcParams.update({'font.size': 24})  # must set in top

            plt.plot(truth, label='Truth', linewidth=3)
            plt.plot(predictions, label='Prediction', linewidth=3)
            plt.title("Ground Truth vs Prediction for Hourly Heat Consumption")
            plt.legend()
            ax.tick_params(axis='both', which='major', labelsize=14)
            ax.tick_params(axis='both', which='minor', labelsize=14)
            ax.set_xlabel('Datetime', fontdict={'fontsize': 16})
            ax.set_ylabel('Hourly Consumption (kW)', fontdict={'fontsize': 16})
            figure = plt.gcf()  # get current figure

            figure.set_size_inches(21, 9)
            plt.savefig(save_base_path + "/Individual_TestData" + '/Forecast_GroundTruth_TestData' + str(i) + '.png', bbox_inches='tight', dpi=600)
            plt.savefig(save_base_path + "/Individual_TestData" + '/Forecast_GroundTruth_TestData' + str(i) + '.pdf', bbox_inches='tight', dpi=600)
            plt.close()


def plot_forecast_timeseries(config_data, result_array):
    """
    Write an interactive whole-series plot of predicted vs ground-truth hourly
    consumption across the full test set to an HTML file.
    :param config_data: config object (used for the DMA name in the output filename)
    :param result_array: array of [datetime (ns), ground_truth, prediction] per sample
    """
    df = pd.DataFrame(np.reshape(result_array, [-1, 3]), columns=['datetime', 'ground_truth', 'prediction'])
    df['datetime'] = pd.to_datetime(df.iloc[:, 0], unit='ns')

    fig = px.line(df, x='datetime', y=['ground_truth', 'prediction'], render_mode='svg')
    fig.update_layout(title="Hourly Consumption - Predicted vs Ground Truth",
                      xaxis_title='Datetime', yaxis_title='Hourly Consumption', font=dict(size=24))
    fig.write_html(run_output_dir() + PREDICTIONS_DIRNAME
                   + f"/{config_data.paths.dma}_Forecast_GroundTruth.html")



def per_sample_errors(truth: np.array, prediction: np.array) -> (np.array, np.array, np.array, np.array):
    """Per-sample MSE, RMSE, MAPE and MAE for aligned ``[samples, horizon]`` arrays.

    Each row is one horizon-length forecast; the metric is computed within each
    row. Shared by the daily test report and the aggregate (test/validation)
    metric summaries.

    :return: tuple of ``(mse, rmse, mape, mae)`` arrays, one value per sample.
    """
    n = len(truth)
    mse, rmse, mape, mae = np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
    for i in range(0, n):
        mse[i] = metrics.mean_squared_error(truth[i], prediction[i])
        rmse[i] = metrics.root_mean_squared_error(truth[i], prediction[i])
        mape[i] = metrics.mean_absolute_percentage_error(truth[i], prediction[i])
        mae[i] = metrics.mean_absolute_error(truth[i], prediction[i])
    return mse, rmse, mape, mae


def calculate_overall_metrics(truth: np.array, prediction: np.array) -> [float, float, float, float]:
    """Mean ``[RMSE, MAPE, MAE, MSE]`` over samples for aligned ``[samples, horizon]`` arrays.

    Timestamp-independent, so it scores either the chronological test window or
    the shuffled validation split.
    """
    mse, rmse, mape, mae = per_sample_errors(truth, prediction)
    return [rmse.mean(), mape.mean(), mae.mean(), mse.mean()]


def calculate_generic_metrics_for_test_data_daily(results_array: np.array) -> [float, float, float, float]:
    """
    Compute per-day (per-sample) error metrics over the test set, then average them.

    For each sample computes MSE, RMSE, MAPE, and MAE between truth and
    prediction, writes the per-day table plus one plot per metric, and returns
    the dataset-level means.

    :param results_array: array of [datetime, ground_truth, prediction] per sample.
    :return: [mean RMSE, mean MAPE, mean MAE, mean MSE] over the test set.
    """

    mse_day, rmse_day, mape_day, mae_day = per_sample_errors(results_array[:, :, 1], results_array[:, :, 2])
    day = results_array[:, 0, 0]  # each sample's first-hour timestamp, for labelling the daily table
    err_df = pd.DataFrame(list(zip(day, mse_day, rmse_day, mape_day, mae_day)), columns=['date', 'MSE', 'RMSE', 'MAPE', 'MAE'])
    err_df['date'] = pd.to_datetime(err_df['date']).dt.date
    err_df.to_csv(run_output_dir() + METRICS_DIRNAME + '/' + "err_df_daily.csv", )

    for i in range(1, err_df.shape[1]):
        fig, ax = plt.subplots()
        plt.rcParams.update({'font.size': 24})  # must set in top

        plt.plot(err_df.iloc[:, 0], err_df.iloc[:, i], label=err_df.columns[i], linewidth=3)
        plt.title("Daily Forecasting Error over the Test Period")
        plt.legend()
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.tick_params(axis='both', which='minor', labelsize=14)
        ax.set_xlabel('Datetime', fontdict={'fontsize': 16})
        ax.set_ylabel(err_df.columns[i], fontdict={'fontsize': 16})
        figure = plt.gcf()  # get current figure

        figure.set_size_inches(21, 9)
        plt.savefig(run_output_dir() + METRICS_DIRNAME + '/' + err_df.columns[i] + '_daily_testdata.png', bbox_inches='tight', dpi=600)
        plt.savefig(run_output_dir() + METRICS_DIRNAME + '/'  + err_df.columns[i] + '_daily_testdata.pdf', bbox_inches='tight', dpi=600)
        plt.close()

    mse_day, rmse_day, mape_day, mae_day = mse_day.mean(), rmse_day.mean(), mape_day.mean(), mae_day.mean()

    return [rmse_day, mape_day, mae_day, mse_day]




def create_folders_for_results():
    """Create the per-run output subdirectories (test data, metrics, plots, model, predictions)."""
    base = run_output_dir()
    Path(base + TEST_DATA_DIRNAME).mkdir(parents=True, exist_ok=True)
    Path(base + METRICS_DIRNAME).mkdir(parents=True, exist_ok=True)
    Path(base + PREDICTIONS_DIRNAME).mkdir(parents=True, exist_ok=True)
    Path(base + MODEL_DIRNAME).mkdir(parents=True, exist_ok=True)
    Path(base + TRUTH_PREDICTION_DIRNAME).mkdir(parents=True, exist_ok=True)


def write_metric_logs_to_file(metric_df, split_label='Test', file_prefix=''):
    """Write the overall mean metrics to a human-readable log and CSV summaries.

    :param metric_df: single-column frame indexed by ['rmse', 'mape', 'mae', 'mse'].
    :param split_label: split name shown in the log header (e.g. ``'Test'`` / ``'Validation'``).
    :param file_prefix: prepended to the output filenames so different splits do
        not overwrite each other (``''`` for test, ``'Validation_'`` for validation).
    :return: the same ``metric_df``, unchanged.
    """
    metrics_dir = run_output_dir() + METRICS_DIRNAME
    file_name = metrics_dir + '/' + file_prefix + 'OverallMetrics.log'

    with open(file_name, "a") as file:
        lines = [f"Overall Mean Metrics of Virtual Meter ({split_label})",
                 "\n---------------------------------------------------------",
                 f"\nAverage RMSE Error: {metric_df.loc['rmse', 0].item():^8}",
                 f"\nAverage MAPE Error: {metric_df.loc['mape', 0].item():^8}",
                 f"\nAverage MAE Error: {metric_df.loc['mae', 0].item():^8}",
                 f"\nAverage MSE Score: {metric_df.loc['mse', 0].item():^8}"]
        file.writelines(lines)

    values = np.array([metric_df.loc['rmse', 0].item(), metric_df.loc['mape', 0].item(), metric_df.loc['mae', 0].item(), metric_df.loc['mse', 0].item()]).reshape([1,-1])
    metrics_final_df = pd.DataFrame(values, columns=['rmse', 'mape', 'mae', 'mse'], index=['metric'])

    metrics_final_df.to_csv(metrics_dir + '/' + file_prefix + 'OverallMetrics.csv')
    metric_df.to_csv(metrics_dir + '/' + file_prefix + 'metrics_df.csv')

    return metric_df




def process_error_metrics(result_array: np.array) -> (pd.DataFrame, pd.DataFrame):
    """
    Score the forecast over the test set and persist the metric artifacts.

    Writes the datetime-indexed truth/prediction table to CSV, computes the
    daily error metrics, and logs the overall means to disk.

    :param result_array: array of [datetime, ground_truth, prediction] per sample.
    :return: metric frame indexed by ['rmse', 'mape', 'mae', 'mse'].
    """

    print(f"Calculating Error Metrics")

    # Flatten samples into a datetime-indexed truth/prediction table.
    truth_prediction = pd.DataFrame(np.concatenate(result_array))
    truth_prediction.index = pd.to_datetime(truth_prediction[0], utc=True)
    truth_prediction = truth_prediction.drop(columns=[0])
    truth_prediction.columns = [0, 1]

    truth_prediction.to_csv(run_output_dir() + TRUTH_PREDICTION_DIRNAME + "/Truth_Prediction.csv")

    metric_df = calculate_generic_metrics_for_test_data_daily(result_array)  # per-day metrics, averaged

    metric_df = pd.DataFrame(metric_df, index=['rmse', 'mape', 'mae', 'mse'])
    metric_df = write_metric_logs_to_file(metric_df)  # save the overall metrics to a log file

    return metric_df


def process_validation_metrics(truth: np.array, prediction: np.array) -> pd.DataFrame:
    """
    Score the best model on the validation split and persist the aggregate metrics.

    Validation samples are shuffled during the split, so only the dataset-level
    means (RMSE/MAPE/MAE/MSE) are meaningful; unlike the test set there is no
    chronological daily breakdown. Results are written to ``Validation_*``
    artifacts so they sit alongside — and do not overwrite — the test metrics.

    :param truth: inverse-scaled validation ground truth, shape [samples, horizon].
    :param prediction: inverse-scaled validation forecast, shape [samples, horizon].
    :return: metric frame indexed by ['rmse', 'mape', 'mae', 'mse'].
    """

    metric_df = calculate_overall_metrics(truth, prediction)
    metric_df = pd.DataFrame(metric_df, index=['rmse', 'mape', 'mae', 'mse'])
    metric_df = write_metric_logs_to_file(metric_df, split_label='Validation', file_prefix='Validation_')

    return metric_df





def plot_loss_curves(training_loss, validation_loss) -> None:
    """Save the training/validation loss curves as PNG and PDF to the run directory."""
    save_path = run_output_dir()
    epochs = np.arange(1, len(training_loss) + 1)
    fig_losscurves, ax = plt.subplots()
    # Skip the first two epochs so the initial loss spike doesn't flatten the curve.
    plt.plot(epochs[2:], training_loss[2:], label='Training Loss')
    plt.plot(epochs[2:], validation_loss[2:], label='Validation Loss')
    plt.title("Loss Curves")
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.grid(True, linestyle=':')
    plt.tight_layout()
    figure = plt.gcf()  # get current figure
    figure.set_size_inches(21, 9)
    plt.savefig(save_path + "LossCurves" + ".png", bbox_inches='tight', dpi=600)
    plt.savefig(save_path + "LossCurves" + ".pdf", bbox_inches='tight', dpi=600)
    plt.close()

    return fig_losscurves

