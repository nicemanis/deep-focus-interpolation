import os
import time
import pickle
import datetime

from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping

from dfi.test import test
from dfi.data import load_data
from dfi.utils import reset_seeds


def train(model, hparams, session_name: str = None):
    reset_seeds(hparams.training.seed)

    start_time = datetime.datetime.now()

    if session_name is None:
        session_name = str(hparams.data.dataset)\
                       + "_" + hparams.training.loss\
                       + "_" + start_time.strftime("%Y%m%d_%H%M%S")

    # Load data
    x, y = load_data(hparams, subset="train")

    # Initialize callbacks
    tensorboard = TensorBoard(log_dir=os.path.join(hparams.io.logs_dir, session_name))
    checkpointer = ModelCheckpoint(filepath=os.path.join(hparams.io.weights_dir, session_name + ".h5"),
                                   verbose=1, save_best_only=True)
    earlystopping = EarlyStopping(monitor="val_loss", patience=hparams.training.patience)

    # Train model
    start = time.time()
    history = model.fit(
        x=x,
        y=y,
        batch_size=hparams.training.batch_size,
        epochs=hparams.training.epochs,
        validation_split=hparams.training.validation_split,
        callbacks=[tensorboard, checkpointer, earlystopping]
    )
    end = time.time()

    # Save model
    model.save(os.path.join(hparams.io.models_dir, session_name))

    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)

    # Save history
    history_dict = {
        "hparams": hparams,
        "epoch": history.epoch,
        "history": history.history,
        "params": history.params,
        "start_time": start_time.strftime("%d.%m.%Y %H:%M:%S"),
        "time_elapsed": "{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds)
    }

    with open(os.path.join(hparams.io.history_dir, session_name), "wb") as f:
        pickle.dump(history_dict, f)

    del x
    del y

    # Test the model
    results_dict = test(model, hparams)
    with open(os.path.join(hparams.io.test_results_dir, session_name), "wb") as f:
        pickle.dump(results_dict, f)
