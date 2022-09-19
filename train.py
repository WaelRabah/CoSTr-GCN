# from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import LightningLoggerBase
from pytorch_lightning.loggers.logger import rank_zero_experiment
from pytorch_lightning.utilities import rank_zero_only

import torch
from datetime import datetime
from model import CoSTrGCN
import os
import matplotlib.pyplot as plt
import collections
# from data_loaders.data_loader import init_data_loader
from data_loaders.data_loader import load_data_sets
import pytorch_lightning as pl



class History_dict(LightningLoggerBase):
    def __init__(self):
        super().__init__()

        self.history = collections.defaultdict(list)  # copy not necessary here
        # The defaultdict in contrast will simply create any items that you try to access

    @property
    def name(self):
        return "Logger_custom_plot"

    @property
    def version(self):
        return "1.0"

    @property
    @rank_zero_experiment
    def experiment(self):
        # Return the experiment object associated with this logger.
        pass

    @rank_zero_only
    def log_metrics(self, metrics, step):
        # metrics is a dictionary of metric names and values
        # your code to record metrics goes here
        for metric_name, metric_value in metrics.items():
            if metric_name != 'epoch':
                self.history[metric_name].append(metric_value)
            else:  # case epoch. We want to avoid adding multiple times the same. It happens for multiple losses.
                if (not len(self.history['epoch']) or    # len == 0:
                        not self.history['epoch'][-1] == metric_value):  # the last values of epochs is not the one we are currently trying to add.
                    self.history['epoch'].append(metric_value)
                else:
                    pass
        return

    def log_hyperparams(self, params):
        pass


def plot_confusion_matrix(confusion_matrix, labels, filename, eps=1e-5):
    import seaborn as sn
    confusion_matrix_sum_vec = torch.sum(confusion_matrix, dim=1) + eps

    confusion_matrix_percentage = (
        confusion_matrix / confusion_matrix_sum_vec.view(-1, 1))

    plt.figure(figsize=(18, 16))
    sn.heatmap(confusion_matrix_percentage.cpu().numpy(), annot=True,
               cmap="coolwarm", xticklabels=labels, yticklabels=labels)
    plt.savefig(filename, format="eps")


def plot_history(history, title: str) -> None:
    ax1 = plt.subplot(211)
    ax1.set_title("Loss")
    ax1.plot(history["train_loss_epoch"], label="train")
    ax1.plot(history["val_loss_epoch"], label="val")
    plt.xlabel("Epoch")
    ax1.legend()

    ax1 = plt.subplot(212)
    ax1.set_title("Accuracy")
    ax1.plot(history["train_acc_epoch"], label="train")
    ax1.plot(history["val_accuracy_epoch"], label="val")
    plt.xlabel("Epoch")
    ax1.legend()
    plt.tight_layout()
    plt.show()


def init_model(graph, optimizer_params, labels, num_classes, dropout_rate=.1):
    model = CoSTrGCN(graph, optimizer_params, labels, d_model=128,
                     n_heads=8, num_classes=num_classes, dropout=dropout_rate)

    return model


def train_model(dataset_name="SHREC21"):
    # loading data
    batch_size = 32
    workers = 4
    window_size = 50
    is_gesture_nogesture_model = False
    num_classes = 2 if is_gesture_nogesture_model else 18

    # data_cfg = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.autograd.set_detect_anomaly(True)
    torch.cuda.manual_seed(42)

    def init_data_loader():
        train_loader, val_loader, test_loader, graph = load_data_sets(
            window_size=window_size,
             batch_size=batch_size,
             workers=workers,
             is_segmented=True,
             binary_classes=is_gesture_nogesture_model,
             use_data_aug=True,
             use_aug_by_sw=True
             )

        return train_loader, test_loader, val_loader, graph
    # print("Data Config=",data_cfg)
    train_loader, test_loader, val_loader, adjacency_matrix = init_data_loader()

    labels = [
        "NO GESTURE",
        "RIGHT",
        "KNOB",
        "CROSS",
        "THREE",
        "V",
        "ONE",
        "FOUR",
        "GRAB",
        "DENY",
        "MENU",
        "CIRCLE",
        "TAP",
        "PINCH",
        "LEFT",
        "TWO",
        "OK",
        "EXPAND",
    ]
    d_model = 128
    n_heads = 8
    lr = 1e-3
    betas = (.9, .98)
    epsilon = 1e-9
    weight_decay = 5e-4
    optimizer_params = (lr, betas, epsilon, weight_decay)
    Max_Epochs = 500
    Early_Stopping = 10
    dropout_rate = .3

    time_now = datetime.today().strftime('%Y-%m-%d_%H_%M_%S')
    # folder for saving trained model...
    # change this path to the fold where you want to save your pre-trained model
    model_fold = "./models/CoSTrGCN-{}_{}/".format(
        dataset_name, time_now)
    try:
        os.mkdir(model_fold)
    except:
        pass

    # .........inital model
    print("\ninit model.............")

    model = init_model(adjacency_matrix, optimizer_params, labels, num_classes)
    print(
        f"d_model (the number of expected features in the encoder inputs/outputs):{d_model}")
    print(f"Number of heads :{n_heads}")
    print(f"dropout rate :{dropout_rate}")
    print(f"Learning rate {lr}")
    best_model = f"best_model-{d_model}-{n_heads}"
    checkpoint_callback = ModelCheckpoint(
        monitor="val_accuracy",
        dirpath=model_fold,
        filename=best_model,
        save_top_k=3,
        mode="max",
    )
    early_stop_callback = EarlyStopping(
        monitor="val_accuracy", min_delta=0.00000001, patience=Early_Stopping, verbose=True, mode="max", check_on_train_epoch_end=True)
    # logger = TensorBoardLogger("tb_logs", name=f"CoSTrGCN_Model")
    history = History_dict()
    trainer = pl.Trainer(gpus=1, precision=16, log_every_n_steps=5,
                         max_epochs=Max_Epochs, logger=[history], callbacks=[early_stop_callback, checkpoint_callback])

    # #***********training#***********
    trainer.fit(model, train_loader, val_loader)
    torch.cuda.empty_cache()
    plt.figure(figsize=(15, 8))
    plot_history(history.history, "CoSTrGCN")

    # plot_confusion_matrix(confusion_matrix,labels,'Confusion_matrices/confusion_matrix_{}.eps'.format(dataset_name))
    model = CoSTrGCN.load_from_checkpoint(checkpoint_path=checkpoint_callback.best_model_path, adjacency_matrix=adjacency_matrix,
                                          optimizer_params=optimizer_params, labels=labels, d_model=128, n_heads=8, num_classes=num_classes, dropout=dropout_rate)
    model.eval()
    test_metrics = trainer.test(model, dataloaders=test_loader)

    print(test_metrics)


if __name__ == "__main__":
    train_model()
