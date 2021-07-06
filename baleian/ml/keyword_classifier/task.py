import os
import argparse
import numpy as np
import tensorflow as tf
import tensorflowjs as tfjs
from sklearn.model_selection import StratifiedKFold

from .util import load_data, get_tensorboard_summaries
from .model import create_training_model, create_ensemble_model


def k_fold_data_generator(features, labels, n_splits=1, shuffle=False, seed=None):
    k_fold = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=seed)
    for train_index, validation_index in k_fold.split(features, labels):
        train_data = (features[train_index], labels[train_index])
        validation_data = (features[validation_index], labels[validation_index])
        yield train_data, validation_data


class CheckpointCallback(tf.keras.callbacks.Callback):
    def __init__(self, checkpoint_manager):
        super(CheckpointCallback, self).__init__()
        self.checkpoint_manager = checkpoint_manager

    def on_epoch_end(self, epoch, logs=None):
        self.checkpoint_manager.save(checkpoint_number=epoch)


def train(model, train_data, validation_data, num_epochs, batch_size, checkpoint_path, tensorboard_path, verbose=2):
    train_dataset = tf.data.Dataset.from_tensor_slices(train_data) \
        .shuffle(buffer_size=len(train_data[0])) \
        .repeat(num_epochs) \
        .batch(batch_size)

    validation_dataset = tf.data.Dataset.from_tensor_slices(validation_data) \
        .repeat(num_epochs) \
        .batch(batch_size)

    checkpoint = tf.train.Checkpoint(model=model)
    checkpoint_manager = tf.train.CheckpointManager(checkpoint, checkpoint_path, max_to_keep=None)
    restored = checkpoint_manager.restore_or_initialize()
    if restored:
        print('Restored checkpoint:', restored, flush=True)
        latest_epoch = int(restored.split('-')[-1])
        initial_epoch = latest_epoch + 1
    else:
        initial_epoch = 0
    print('initial epoch:', initial_epoch, flush=True)

    checkpoint_cb = CheckpointCallback(checkpoint_manager)
    tensorboard_cb = tf.keras.callbacks.TensorBoard(tensorboard_path, histogram_freq=1, profile_batch=0)

    model.fit(
        train_dataset,
        steps_per_epoch=int(len(train_data[0]) / batch_size) or 1,
        validation_data=validation_dataset,
        validation_steps=int(len(validation_data[0]) / batch_size) or 1,
        initial_epoch=initial_epoch,
        epochs=num_epochs,
        callbacks=[checkpoint_cb, tensorboard_cb],
        verbose=verbose)


def get_summary_values(tensorboard_path, tag, default_value=None):
    summaries = {}
    for s in get_tensorboard_summaries(tensorboard_path):
        if s['tag'] == tag:
            summaries[s['step']] = s['value']
    values = [default_value] * (max(summaries.keys()) + 1)
    for i, v in summaries.items():
        values[i] = v
    return values


def main(train_data_path, num_epochs, num_folds, batch_size, job_dir):
    checkpoint_path = os.path.join(job_dir, 'checkpoint')
    tensorboard_path = os.path.join(job_dir, 'tensorboard')
    model_path = os.path.join(job_dir, 'model')
    tfjs_path = os.path.join(job_dir, 'tfjs')

    features, labels = load_data(train_data_path)
    input_shape = features.shape[1:]
    num_class = labels.max() + 1
    
    data_generator = k_fold_data_generator(features, labels, n_splits=num_folds, shuffle=True, seed=123)

    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        models = []
        # Train models for each k_fold splitted dataset
        for index, (train_data, validation_data) in enumerate(data_generator):
            model = create_training_model(input_shape, num_class, name=f'fold_{index}')
            models.append(model)
            print(model.name + ' training start.', flush=True)
            train(model, train_data, validation_data,
                  num_epochs=num_epochs,
                  batch_size=batch_size,
                  checkpoint_path=os.path.join(checkpoint_path, model.name),
                  tensorboard_path=os.path.join(tensorboard_path, model.name),
                  verbose=1)
            print(model.name + ' training finished.', flush=True)

        # Get best epoch of average val_loss for each models
        val_losses = [
            get_summary_values(
                os.path.join(tensorboard_path, f'{model.name}/validation'),
                tag='epoch_loss',
                default_value=float('inf')
            )
            for model in models
        ]
        avg_val_losses = np.array(val_losses).mean(axis=0)
        best_epoch = avg_val_losses.argmin()
        print('avg_val_losses:', avg_val_losses, flush=True)
        print('best_epoch:', best_epoch, flush=True)

        # Restore model weights of best_epoch
        for model in models:
            checkpoint = tf.train.Checkpoint(model=model)
            path = os.path.join(checkpoint_path, f'{model.name}/ckpt-{best_epoch}')
            print(f'Restoring {model.name} checkpoint... {path}')
            checkpoint.restore(path)
        
        # Create ensemble model
        ensemble = create_ensemble_model(input_shape, num_class, models, name='ensemble')

        summary_writer = tf.summary.create_file_writer(os.path.join(tensorboard_path, ensemble.name))

        # Evaluate ensemble model
        tf.summary.trace_on()
        print('evaluating ensemble model...', flush=True)
        loss, accuracy = ensemble.evaluate(features, labels, batch_size=batch_size, verbose=1)
        print('evaluated loss:', loss, flush=True)
        print('evaluated accuracy:', accuracy, flush=True)
        
        with summary_writer.as_default(step=best_epoch):
            tf.summary.scalar('epoch_loss', loss)
            tf.summary.scalar('epoch_accuracy', accuracy)
            tf.summary.trace_export(ensemble.name)
        tf.summary.trace_off()
    
        # Export ensemble model
        print('Exporting model to ' + model_path, flush=True)
        tf.keras.models.save_model(ensemble, model_path)
        print('Exporting tfjs model to ' + tfjs_path, flush=True)
        tfjs.converters.save_keras_model(ensemble, tfjs_path)
        print('Exporting completed.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-data-path', required=True)
    parser.add_argument('--num-epochs', type=int, default=1)
    parser.add_argument('--num-folds', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=1024*10)
    parser.add_argument('--job-dir', required=True)
    args = parser.parse_args()
    main(train_data_path=args.train_data_path,
         num_epochs=args.num_epochs,
         num_folds=args.num_folds,
         batch_size=args.batch_size,
         job_dir=args.job_dir)
