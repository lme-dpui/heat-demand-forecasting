import torch as t
import numpy as np
from typing import Any, Tuple
from src.metrics import Metric
from torch.utils.data.dataloader import DataLoader
import config
from definitions import MODEL_DIRNAME, run_output_dir
import os
import time
import copy


class Trainer:

    def __init__(self,
                 params: config.WaveletConfig,
                 model: t.nn.Module,  # Model to be trained.
                 loss_fn=None,  # loss function
                 optim: t.optim.Optimizer = None,  # Optimizer
                 scheduler=None,  # LR scheduler stepped on the validation loss each epoch
                 training_data: DataLoader = None,  # Training data set
                 validation_data: DataLoader = None,  # Validation (or test) data set
                 early_stopping_patience: int = -1,
                 cuda: bool = True):  # The patience for early stopping

        print(f"Initialising Trainer...")
        self._params = params
        self._model = model
        self._best_model = None
        self._loss_fn = loss_fn

        self._optim = optim
        self._scheduler = scheduler
        self._training_data = training_data
        self._validation_data = validation_data
        self._patience = early_stopping_patience
        self._cuda = cuda

        self.best_validation_score: float = np.inf
        self.best_epoch: int = 0

        self.model_save_path = run_output_dir() + MODEL_DIRNAME
        if os.path.exists(self.model_save_path) is False:
            os.mkdir(self.model_save_path)

        self.training_loss = Metric()
        self.validation_loss = Metric()
        if cuda:
            self._model = model.cuda()
            self._loss_fn = self._loss_fn.cuda()


    def get_best_model(self):
        return self._best_model

    def save_checkpoint(self):
        t.save({'state_dict': self._model.state_dict(),
                'optimizer_state_dict': self._optim.state_dict(),
                'val_loss': self.best_validation_score}, self.model_save_path + '/best_model.ckp')

    def restore_checkpoint(self):
        print(f"Resuming training from checkpoint epoch...")
        ckp = t.load(self.model_save_path + '/best_model.ckp', 'cuda' if self._cuda else None)
        self._model.load_state_dict(ckp['state_dict'])
        self._optim.load_state_dict(ckp['optimizer_state_dict'])

    def save_onnx(self, fn):
        m = self._model.cpu()
        m.eval()
        x = t.randn(1, 3, 300, 300, requires_grad=True)
        y = self._model(x)
        t.onnx.export(m,  # model being run
                      x,  # model input (or a tuple for multiple inputs)
                      fn,  # where to save the model (can be a file or file-like object)
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=10,  # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=['input'],  # the model's input names
                      output_names=['output'],  # the model's output names
                      dynamic_axes={'input': {0: 'batch_size'},  # variable lenght axes
                                    'output': {0: 'batch_size'}})

    def _forward_loss(self, x: Any, y: Any) -> Tuple[Any, Any]:

        prediction = self._model(x)
        loss = self._loss_fn(prediction, y)

        return loss, prediction

    def train_step(self, x: Any, y: Any) -> (float, Any):

        self._optim.zero_grad(set_to_none=True)

        loss, prediction = self._forward_loss(x, y)

        loss.backward()
        self._optim.step()

        return float(loss), prediction

    def validation_step(self, x: Any, y: Any) -> (float, Any):

        loss, prediction = self._forward_loss(x, y)

        return float(loss), prediction

    def _run_epoch(self, data: DataLoader, train: bool) -> float:

        self._model.train(train)  # train=True -> training mode, False -> eval mode
        step = self.train_step if train else self.validation_step

        total_loss = 0
        with t.set_grad_enabled(train):
            for i, (x, y) in enumerate(data):
                if self._cuda is True:
                    x, y = x.cuda(), y.cuda()

                batch_loss, _ = step(x, y)
                total_loss = total_loss + batch_loss

        return total_loss / (i + 1)

    def fit(self, epochs: int = -1, restore_epoch: int = None):
        assert self._patience > 0 or epochs > 0


        print(f"Training the model...")
        if restore_epoch != -1:
            self.restore_checkpoint()
            epoch_number, patience = restore_epoch, 0  # epoch counter, early stopping criteria
        else:
            epoch_number, patience = 0, 0  # epoch counter, early stopping criteria

        self.best_validation_score = np.inf

        training_start_time = time.time()
        while epoch_number <= epochs:

            train_loss = self._run_epoch(self._training_data, train=True)
            validation_loss = self._run_epoch(self._validation_data, train=False)

            if self._scheduler is not None:
                self._scheduler.step(validation_loss)  # decay the LR on a validation-loss plateau

            if validation_loss < self.best_validation_score:
                self.best_validation_score = validation_loss
                self.best_epoch = epoch_number
                #self.save_checkpoint()
                self._best_model = copy.deepcopy(self._model)
                patience = 0
            else:
                patience += 1

            if patience >= self._patience:
                print(f"Reached Early Stopping Criteria")
                break

            if epoch_number % 10 == 0:
                summary = ', '.join([f"[Epoch: {epoch_number}/{epochs}]",
                                     f"Validation Accuracy: {validation_loss: 0.4f}",
                                     f"Train Accuracy: {train_loss: 0.4f}",
                                     f"Patience: {patience}"])
                print(summary)

            self.training_loss.update(train_loss, epoch_number+1)
            self.validation_loss.update(validation_loss, epoch_number+1)
            epoch_number += 1


        self.total_training_time = (time.time() - training_start_time)/60

        print(f"Best Epoch: {self.best_epoch:4}")



