# SlidingWindowGenerator
- based on tensorflow v2.3.0
    - use [timeseries_dataset_from_array function](https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/timeseries_dataset_from_array) which was introduced in tf v2.3.0 
- This module converts time series data from dataframe type to sliding window type
    - to use as input in RNN based layer
- This module was based on [tensorflow official docs](https://www.tensorflow.org/tutorials/structured_data/time_series#data_windowing), just aggregate some functions and add small tuning to use it more efficiently.
    - to make it possible to control batch_size, sequence_stride_size and shuffle more freely.
- Module is in `./src/sliding_window_generator.py` and the example of how to use it is in `./src/example.ipynb`