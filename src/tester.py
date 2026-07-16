import torch as t
import numpy as np
import os
import pandas as pd

from definitions import MODEL_DIRNAME, TRUTH_PREDICTION_DIRNAME, run_output_dir


class Tester:

    def __init__(self,
                 params,
                 model,  # Model to be trained.
                 crit,  # Loss function
                 optim=None,  # Optimizer
                 test_data=None,  # Training data set
                 scaler=None,
                 cuda=True):

        self._params = params
        self._model = model
        self._crit = crit
        self._optim = optim
        self._test_data = test_data
        self._scaler = scaler
        self._cuda = cuda

        if cuda:
            self._model = model.cuda()
            self._crit = crit.cuda()

    def restore_checkpoint(self):
        ckp_path = run_output_dir() + MODEL_DIRNAME + '/best_model.ckp'
        assert os.path.exists(ckp_path), "AssertionError: No file to load. Model file is missing."
        ckp = t.load(ckp_path, 'cuda' if self._cuda else None)
        self._model.load_state_dict(ckp['state_dict'])
        self._optim.load_state_dict(ckp['optimizer_state_dict'])

    def evaluate(self, data_loader):
        """Run the model over ``data_loader`` and return inverse-scaled forecasts.

        Both returned arrays have shape ``[samples, prediction_horizon]`` and are
        on the original consumption scale (undoing the min-max scaling applied
        during training), so they can be scored directly. Used for both the test
        and validation splits.

        :return: tuple ``(ground_truth, prediction)``.
        """
        predictions = []
        ground_truth = []

        self._model.eval()
        with t.no_grad():
            for x, y in data_loader:
                if self._cuda:
                    x = x.cuda()

                y_hat = self._model(x)

                ground_truth.extend(y.detach().cpu().numpy())
                predictions.extend(y_hat.detach().cpu().numpy())

        data_span = self._scaler.data_max_[0] - self._scaler.data_min_[0]
        data_min = self._scaler.data_min_[0]

        ground_truth = np.array(ground_truth) * data_span + data_min
        predictions = np.array(predictions) * data_span + data_min

        return ground_truth, predictions

    def test(self, test_timeframe):

        ground_truth, predictions = self.evaluate(self._test_data)

        result_array = np.empty([len(ground_truth), self._params.time.prediction_horizon, 3])
        result_array[:, :, 1] = ground_truth
        result_array[:, :, 2] = predictions

        test_timeframe = test_timeframe[-len(ground_truth):]
        for i in range(0, len(ground_truth)):
            result_array[i, :, 0] = pd.to_datetime(test_timeframe[i, :])

        reshape_arr = result_array.reshape(result_array.shape[0], result_array.shape[1] * result_array.shape[2])
        np.savetxt(run_output_dir() + TRUTH_PREDICTION_DIRNAME + "/Truth_Prediction.npy", reshape_arr, delimiter=",", fmt="%f")

        return result_array

