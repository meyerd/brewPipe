# BrewPipe

BrewPipe is supposed to be a framework for quick prototypes in data processing.
The idea is to have a data processing pipeline, that is general enough to 
accomodate a wide variety of tasks in crunching data, but can be specialized 
to certain models or data shapes.

The general idea to have processing blocks of the follwing types

* *Data*: The input data of a specific type. For example CSV data, that will be
  read into for example pandas data frames for further processing.
* *Preprocessing*: For cleaning up the data, if that is not already handled in
  the data loading phase. A simple numpy caching can be easily implemented by 
  using the lazy dataframe capability and the `numpy_null` preprocessor, that
  does nothing to the data.
* *Model*: Models for specific data and/or data processing tasks. These models
  can for example be used to learn a specific neural net model for some part of
  the data and then be used to output newly learned results.
* *Output*: Output formatters to generate the output of models in a specific 
  data format, that is for example required for submissions to specific 
  data processing challenges.

In general all models should support the input and output of so called _lazy dataframes_.
The special particularity of those dataframes is that the data will be cached (if
implemented) and only evaluated if the input to the block changed. Therefore,
certain blocks can be evaluated and the preprocessing of the data does not neccessarly
have to happen as this data is already available in the cache. Also the data 
processing of a lazy dataframe is delayed until the very last moment just before
the data is really needed.

## Currently supported

* Simple numpy model
* TensorFlow
* Pandas data loading
* Lazy Dataframes

## Examples

This example was written for the 
[Winton Stock Market Challenge](https://www.kaggle.com/c/the-winton-stock-market-challenge) 
and consists of two very simple models to predict stock data.

To run the experiments, download the data from the stockmarket challenge
website and put it in the `data/` directory.

Then run either `scratchpad/run_leastsquares_winton.py`, which will learn a 
least squares model of the intraday trading data for prediction using tensorflow
and output the predicted data in the format ready for submission to the challenge.

The other very simple model is to learn the mean and variance of certain variables
that are to predicted by averaging and then sample from a normal distribution for
prediction data. To run this example run `scratchpad/run_mean_variance_winton.py`

