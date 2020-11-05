import numpy as np
import tensorflow as tf


class SlidingWindowGenerator:
    def __init__(self, input_width, label_width, shift,
                 train_df, val_df, test_df,
                 label_columns=None):
        # Store raw data with dataframe type
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df

        # Work out the label column indices.
        self.label_columns = label_columns
        if label_columns is not None:
            self.label_columns_indices = {name: i for i, name in
                                          enumerate(label_columns)}
        self.column_indices = {name: i for i, name in
                               enumerate(train_df.columns)}

        # Work out the window parameters.
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift  # indicates the label offset far from input

        self.total_window_size = input_width + shift

        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[
            self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[
            self.labels_slice]

    def __repr__(self):
        return '\n'.join([
            f'Total window size: {self.total_window_size}',
            f'Input indices: {self.input_indices}',
            f'Label indices: {self.label_indices}',
            f'Label column name(s): {self.label_columns}'])

    def train(self, sequence_stride=1, shuffle=True, batch_size=32):
        """
        make train time series dataset
        :param sequence_stride: int
        :param shuffle: boolean
        :param batch_size: int
        :return: time_series data
        """
        return self.make_dataset(self.train_df,
                                 sequence_stride=sequence_stride,
                                 shuffle=shuffle, batch_size=batch_size)

    def val(self, sequence_stride=1, shuffle=False, batch_size=32):
        """
        make validation time series dataset
        :param sequence_stride: int
        :param shuffle: boolean
        :param batch_size: int
        :return: time_series data
        """
        return self.make_dataset(self.val_df, sequence_stride=sequence_stride,
                                 shuffle=shuffle, batch_size=batch_size)

    def test(self, sequence_stride=1, shuffle=False, batch_size=32):
        """
        make test time series dataset
        :param sequence_stride: int
        :param shuffle: boolean
        :param batch_size: int
        :return: time_series data
        """
        return self.make_dataset(self.test_df, sequence_stride=sequence_stride,
                                 shuffle=shuffle, batch_size=batch_size)

    def example(self, sequence_stride=1, shuffle=True, batch_size=32):
        """
        Get and cache an example batch of `inputs, labels` for checking shape
        """
        result = getattr(self, '_example', None)
        if result is None:
            # No example batch was found, so get one from the `.train` dataset
            result = next(iter(
                self.train(sequence_stride=sequence_stride, shuffle=shuffle,
                           batch_size=batch_size)))
            # And cache it for next time
            self._example = result
        return result

    def split_window(self, features):
        """
        input 을 input 용 column 들로 이루어진 data 와
        label 용 column 들로 이루어진 data 로 분리
        """
        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :]
        if self.label_columns is not None:
            labels = tf.stack(
                [labels[:, :, self.column_indices[name]] for name in
                 self.label_columns],
                axis=-1)

        # Slicing doesn't preserve static shape information, so set the shapes
        # manually. This way the `tf.data.Datasets` are easier to inspect.
        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])

        return inputs, labels

    def make_dataset(self, data, sequence_stride=1, shuffle=True,
                     batch_size=32):
        data = np.array(data, dtype=np.float32)
        ds = tf.keras.preprocessing.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=sequence_stride,
            shuffle=shuffle,
            batch_size=batch_size, )

        ds = ds.map(self.split_window)

        return ds
