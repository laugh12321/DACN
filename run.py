"""
Created on Jan 29, 2021

@file: runner.py
@desc: Run experiments given set of hyperparameters for the unmixing problem.
@author: laugh12321
@contact: laugh12321@vip.qq.com
"""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from config.get_config import get_config

from src.model import enums
from src.utils.utils import parse_train_size, subsample_test_set
from src.utils import prepare_data, artifacts_reporter
from src.model import evaluate_unmixing, train_unmixing
from src.model.models import rnn_supervised, pixel_based_cnn, \
    pixel_based_fnnc, pixel_based_dacn

# Literature hyperparameters settings:
LEARNING_RATES = {
    rnn_supervised.__name__: 0.001,

    pixel_based_cnn.__name__: 0.01,

    pixel_based_fnnc.__name__: 0.0001,

    pixel_based_dacn.__name__: 3e-3
}


def run_experiments(*,
                    data_file_path: str,
                    ground_truth_path: str = None,
                    train_size: int or float,
                    val_size: float = 0.1,
                    sub_test_size: int = None,
                    n_runs: int = 4,
                    model_name: str,
                    dest_path: str = None,
                    sample_size: int,
                    n_classes: int,
                    lr: float = None,
                    batch_size: int = 256,
                    epochs: int = 100,
                    verbose: int = 1,
                    shuffle: bool = True,
                    patience: int = 15):
    """
    Function for running experiments on unmixing given a set of hyper parameters.

    :param data_file_path: Path to the data file. Supported types are: .npy.
    :param ground_truth_path: Path to the ground-truth data file.
    :param train_size: If float, should be between 0.0 and 1.0.
        If int, specifies the number of samples in the training set.
        Defaults to 0.8
    :type train_size: Union[int, float]
    :param val_size: Should be between 0.0 and 1.0. Represents the
        percentage of samples from the training set to be
        extracted as a validation set.
        Defaults to 0.1.
    :param sub_test_size: Number of pixels to subsample the test set
        instead of performing the inference on all
        samples that are not in the training set.
    :param n_runs: Number of total experiment runs.
    :param model_name: Name of the model, it serves as a key in the
        dictionary holding all functions returning models.
    :param dest_path: Path to where all experiment runs will be saved as
        subdirectories in this directory.
    :param sample_size: Size of the input sample.
    :param n_classes: Number of classes.
    :param lr: Learning rate for the model, i.e., regulates
        the size of the step in the gradient descent process.
    :param batch_size: Size of the batch used in training phase,
        it is the size of samples per gradient step.
    :param epochs: Number of epochs for model to train.
    :param verbose: Verbosity mode used in training, (0, 1 or 2).
    :param shuffle: Boolean indicating whether to shuffle datasets.
    :param patience: Number of epochs without improvement in order to
        stop the training phase.
    """
    for experiment_id in range(n_runs):
        experiment_dest_path = os.path.join(dest_path,
                                            '{}_{}'.format(enums.Experiment.EXPERIMENT, str(experiment_id)))
        os.makedirs(experiment_dest_path, exist_ok=True)

        # Apply default literature hyper parameters:
        if lr is None and model_name in LEARNING_RATES:
            lr = LEARNING_RATES[model_name]

        # Prepare data for unmixing:
        data = prepare_data.main(data_file_path=data_file_path,
                                 ground_truth_path=ground_truth_path,
                                 train_size=parse_train_size(train_size),
                                 val_size=val_size,
                                 seed=experiment_id)
        # Subsample the test set to constitute a constant size:                        
        if sub_test_size is not None:
            subsample_test_set(data[enums.Dataset.TEST], sub_test_size)
        
        # Train the model:
        train_unmixing.train(model_name=model_name,
                             dest_path=experiment_dest_path,
                             data=data,
                             sample_size=sample_size,
                             n_classes=n_classes,
                             lr=lr,
                             batch_size=batch_size,
                             epochs=epochs,
                             verbose=verbose,
                             shuffle=shuffle,
                             patience=patience,
                             seed=experiment_id)
        # Evaluate the model:
        evaluate_unmixing.evaluate(
            model_name=model_name,
            data=data,
            dest_path=experiment_dest_path,
            batch_size=batch_size)
        tf.keras.backend.clear_session()

    artifacts_reporter.collect_artifacts_report(
        experiments_path=dest_path,
        dest_path=dest_path)


if __name__ == '__main__':
    args = get_config(filename='./config/config.json')

    for model_name in args.model_names:
        for data_name in args.dataset:
            dest_path = os.path.join(args.save_path,
                                    '{}_{}'.format(str(model_name), str(data_name)))

            base_path = os.path.join(args.path, data_name)
            data_file_path = os.path.join(base_path, data_name + '.npy')
            ground_truth_path = os.path.join(base_path, data_name + '_gt.npy')

            if data_name == 'urban':
                sample_size, n_classes = 162, 6
            else:
                sample_size, n_classes = 157, 4

            run_experiments(data_file_path=data_file_path,
                            ground_truth_path=ground_truth_path,
                            dest_path=dest_path,
                            train_size=args.train_size,
                            val_size=args.val_size,
                            model_name=model_name,
                            sample_size=sample_size,
                            n_classes=n_classes,
                            batch_size=args.batch_size,
                            epochs=args.epochs,
                            verbose=args.verbose,
                            patience=args.patience,
                            n_runs=args.n_runs)
