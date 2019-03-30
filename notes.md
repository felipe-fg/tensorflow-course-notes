# TensorFlow Crash Course

## Terminology

- label: `y` variable - thing we are predicting
- feature: `x` variable - input variable
- example: `x` vector - particular instance of data
  - labeled example: `(x, y)` - includes both features and the label
  - unlabeled example: `(x, ?)` - contains features but not the label
- model: relationship between features and label
  - training: learning the model
  - inference: apply trained model to unlabeled examples
  - regression: predicts continuous values
  - classification: predicts discrete values

## Linear Regression

- linear: `y = mx + b`
  - `y`: predicted value
  - `m`: slope of the line
  - `x`: input value
  - `b`: y-intercept
- machine learning: `y' = b + w₁x₁`
  - `y'`: predicted label
  - `b`: bias
  - `w₁`: weight of feature 1
  - `x₁`: input of feature 1
- `y' = b + w₁x₁ + w₂x₂ + w₃x₃`

## Traning and loss

- loss: number indicating how bad the model's prediction was on a single example
- squared loss (L₂ loss): `(observation - prediction(x))²` - popular loss function
- mean square error (MSE): average squared loss per example over the whole dataset

## An Iterative Approach

- iterative learning: "hot and cold" game
  - random values for weights and bias
  - compute loss function
  - generate new values for weights and bias
- model converged: loss stops changing (or changes extremely slowly)

## Gradient Descent

- loss curve: loss for all possible values of `w₁`
- gradient descent: better way of finding the convergence point
  - pick a random value for `w₁`
  - compute the gradient of the loss curve at the point
  - use (negative) gradient vector to change `w₁`

## Learning Rate

- learning rate / step size: fraction of gradient vector to use
- too small: learning will take too long
- too big: overshoots the minimum

## Stochastic Gradient Descent

- gradient descent: entire data set
- stochastic gradient descent: single example
- mini-batch stochastic gradient descent: batch of examples (10 ~ 1000)

## Toolkit

- `tf.estimator`: predefined architectures
- `tf.layers / tf.losses / tf.metrics`: reusable libraries
- TensorFlow: lower-level APIs
- steps: total number of training iterations
- batch_size: number of examples (chosen at random) for a single step
- periods: granularity of reporting
- synthetic feature: ratio of two other features

## Peril of Overfitting

- overfitting: low loss during training but does a poor job predicting new data

## Splitting Data

- randomize data before splitting
- do not train on test data
- training set: subset to train a model
- test set: subset to test the model - large enough to yield statistically meaningful results

## Another Partition

- validation set: evaluate results from the training set
- test set: double-check your evaluation after the model has "passed" the validation set

## Feature Engineering

- numeric values: integer and floating-point raw data don't need a special encoding
- string values: define a vocabulary then create a one-hot encoding that represents a given string
- enumerated values: machine learning models typically represent each categorical feature as a separate Boolean value

## Qualities of Good Features

- avoid rarely used discrete feature values
- prefer clear and obvious meanings
- don't mix "magic" values with actual data
- account for upstream instability

## Cleaning Data

- scaling feature values - z score: `scaledvalue = (value - mean) / stddev`
- handling extreme outliers: "cap" or "clip" the maximum value
- binning: bin by quantile - ensures that the number of examples in each bucket is equal
- scrubbing: many examples in data sets are unreliable

## Encoding Nonlinearity

- feature cross: synthetic feature that encodes nonlinearity in the feature space
- `x₃ = x₁x₂`
- `y' = b + w₁x₁ + w₂x₂ + w₃x₃`

## Crossing One-Hot Vectors

- one-hot feature vectors: logical conjunctions

## L₂ Regularization

- regularization: prevent overfitting by penalizing complex models
- empirical risk minimization: minimize loss - `minimize(Loss(Data|Model))`
- structural risk minimization: minimize loss+complexity - `minimize(Loss(Data|Model) + complexity(Model))`
  - loss term: measures how well the model fits the data
  - regularization term: measures model complexity
- model complexity: function of the weights of all the features in the model
- L2 regularization: sum of the squares of all the feature weights
- encourages weight values toward 0 (but not exactly 0)
- encourages the mean of the weights toward 0

## Lambda

- lambda / regularization rate: tune the overall impact of the regularization term
- `minimize(Loss(Data|Model) + λ complexity(Model))`
- increasing the lambda value strengthens the regularization effect
- too high: model will be simple - underfitting the data
- too low: model will be more complex - overfitting the data

## Calculating a Probability

- logistic regression: probability estimate as output
- sigmoid function: output always falls between 0 and 1
- `y' = 1 / (1 + e^(-z))`
  - `y'`: output for a particular example
  - `z`: `b + w₁x₁ + w₂x₂ + ...`

## Model Training

- loss function for linear regression: squared loss
- loss function for logistic regression: log loss
- regularization: extremely important in logistic regression
