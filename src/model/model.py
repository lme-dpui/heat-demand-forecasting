import src.model.WaveletScalogram as WaveletScalogram
import src.feature_names as feature_names
import torch
from src.trainer import Trainer
from src.tester import Tester


class RMSELoss(torch.nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()

    def forward(self, x, y):
        criterion = torch.nn.MSELoss()
        eps = 1e-6
        loss = torch.sqrt(criterion(x, y) + eps)

        return loss


def initiate_training(config_data, train_loader, validation_loader, test_loader, scaler, test_timeframe):

    print(f"Model used: Wavelet Scalogram with VGG")
    model = WaveletScalogram.Wavelet_Scalogram_VGG(config_data.setup.encoding_schema.num_frequencies,  # number of scales - vertical dimension of the image
                                                   config_data.time.historical_values,  # length of historical data - horizontal dimension of the image
                                                   len(feature_names.expand_feature_spec(config_data.variables.features)),  # number of channels in data
                                                   config_data.time.prediction_horizon,
                                                   config_data.setup.architecture)
    print(model)

    if config_data.model_params.loss_function == "MSE":
        loss_function = torch.nn.MSELoss()  # loss function
    elif config_data.model_params.loss_function == "RMSE":
        loss_function = RMSELoss()  # loss function
    elif config_data.model_params.loss_function == "MAE":
        loss_function = torch.nn.L1Loss()

    else:
        raise ValueError(f"Invalid Loss Function - {config_data.model_params.loss_function}")

    optimizer = torch.optim.Adam(model.parameters(), lr=config_data.model_params.lr)  # optimizer
    # Decay the LR whenever the validation loss plateaus (paper Section 2.6.2). Only
    # the decay factor is specified by the paper; the plateau patience/threshold keep
    # PyTorch's defaults.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                           factor=config_data.model_params.lr_scheduler_factor)

    pytorch_total_params = sum(p.numel() for p in model.parameters())
    pytorch_total_params_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total param: {pytorch_total_params},                      Trainable params: {pytorch_total_params_trainable}")

    # Training phase
    use_cuda = torch.cuda.is_available()
    forecast_trainer = Trainer(params=config_data,
                               model=model,  # Model to be trained
                               loss_fn=loss_function,  # loss function
                               optim=optimizer,  # Optimizer
                               scheduler=scheduler,  # LR scheduler stepped on the validation loss
                               training_data=train_loader,  # Training data set
                               validation_data=validation_loader,  # Validation (or test) data set
                               early_stopping_patience=config_data.model_params.patience,
                               cuda=use_cuda)

    forecast_trainer.fit(epochs=config_data.model_params.epoch, restore_epoch=config_data.model_params.restore_epoch)  # actual training process

    forecast_tester = Tester(params=config_data,
                             model=forecast_trainer.get_best_model(),
                             crit=loss_function,
                             optim=optimizer,
                             test_data=test_loader,
                             scaler=scaler,
                             cuda=use_cuda)

    result_array = forecast_tester.test(test_timeframe)
    # Score the same best model on the validation split so its metrics can be
    # reported alongside the test metrics.
    validation_truth, validation_prediction = forecast_tester.evaluate(validation_loader)

    return forecast_trainer, result_array, validation_truth, validation_prediction
