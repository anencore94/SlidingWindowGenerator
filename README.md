# TimeSeriesDataGenerator
- based on tensorflow v2.3.0
    - use [timeseries_dataset_from_array function](https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/timeseries_dataset_from_array) which was introduced in tf v2.3.0 
- This module make time series data from dataframe
    - a.k.a. time window
- This module was based on [tensorflow official docs](https://www.tensorflow.org/tutorials/structured_data/time_series#data_windowing), just aggregate some functions and add small tuning to use it more efficiently.
    - to make it possible to control batch_size, sequence_stride_size and shuffle more freely.
