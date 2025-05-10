---
title: "Beyesian Optimization with Python"
date: "2023-02-27"
author: "Junxiao Guo"
tags: ["machine-learning", "automl"]
excerpt: "Beyesian optimization explained in detail with python implementation."
---


Hyperparameter optimization is a challenging problem of finding an input that results in the minimum or maximum cost of a given objective function.

Bayesian Optimization provides a principled technique based on Bayes Theorem to direct a search of a global optimization problem that is efficient and effective. It works by building a probabilistic model of the objective function, called the surrogate function, that is then searched efficiently with an acquisition function before candidate samples are chosen for evaluation on the real objective function.

Bayesian Optimization is often used in applied machine learning to tune the hyperparameters of a given well-performing model on a validation dataset.

## What is bayesian optimization

Bayesian Optimization is an approach that uses Bayes Theorem to direct the search in order to find the minimum or maximum of an objective function.

It is an approach that is most useful for objective functions that are complex, noisy, and/or expensive to evaluate.

Recall that Bayes Theorem is an approach for calculating the conditional probability of an event:

$$P(A|B) = \frac{P(B|A) * P(A)}{P(B)}$$

We can simplify this calculation by removing the normalizing value of P(B) and describe the conditional probability as a proportional quantity. This is useful as we are not interested in calculating a specific conditional probability, but instead in optimizing a quantity.

$$P(A|B) = P(B|A) * P(A)$$

The conditional probability that we are calculating is referred to generally as the posterior probability; the reverse conditional probability is sometimes referred to as the likelihood, and the marginal probability is referred to as the prior probability; for example:

$$posterior = likelihood * prior$$
This provides a framework that can be used to quantify the beliefs about an unknown objective function given samples from the domain and their evaluation via the objective function.

We can devise specific samples $(x_1, x_2, …, x_n)$ and evaluate them using the objective function f(xi) that returns the cost or outcome for the sample xi. Samples and their outcome are collected sequentially and define our data D, e.g. $D = {x_i, f(x_i), … x_n, f(x_n)}$ and is used to define the prior. The likelihood function is defined as the probability of observing the data given the function $P(D | f)$. This likelihood function will change as more observations are collected.

$$P(f|D) = P(D|f) * P(f)$$
The posterior represents everything we know about the objective function. It is an approximation of the objective function and can be used to estimate the cost of different candidate samples that we may want to evaluate.

In this way, the posterior probability is a surrogate objective function.

- **Surrogate Function**: Bayesian approximation of the objective function that can be sampled efficiently.
The surrogate function gives us an estimate of the objective function, which can be used to direct future sampling. Sampling involves careful use of the posterior in a function known as the “acquisition” function, e.g. for acquiring more samples. We want to use our belief about the objective function to sample the area of the search space that is most likely to pay off, therefore the acquisition will optimize the conditional probability of locations in the search to generate the next sample.

- **Acquisition Function**: Technique by which the posterior is used to select the next sample from the search space.
Once additional samples and their evaluation via the objective function f() have been collected, they are added to data D and the posterior is then updated.

This process is repeated until the extrema of the objective function is located, a good enough result is located, or resources are exhausted.

The Bayesian Optimization algorithm can be summarized as follows:

1. Select a Sample by Optimizing the Acquisition Function.
2. Evaluate the Sample With the Objective Function.
3. Update the Data and, in turn, the Surrogate Function.
4. Go To 1.

## Performing Bayesian Optimization

In this section, we will explore how Bayesian Optimization works by developing an implementation from scratch for a simple one-dimensional test function.

First, we will define the test problem, then how to model the mapping of inputs to outputs with a surrogate function. Next, we will see how the surrogate function can be searched efficiently with an acquisition function before tying all of these elements together into the Bayesian Optimization procedure.

```python
import numpy as np
import math
from matplotlib import pyplot
```

### Problem definition

We will use a multimodal problem with five peaks, calculated as:

$$y = x^2 * cos(3 * \pi * x)^4$$

Where x is a real value in the range $[0,1]$.

We will augment this function by adding Gaussian noise with a mean of zero and a standard deviation of 0.1. This will mean that the real evaluation will have a positive or negative random value added to it, making the function challenging to optimize.

The objective() function below implements this.

```python
# objective function
def objective(x, noise=0.1):
    noise = np.random.normal(loc=0, scale=noise)
    return (x**2 * math.cos(3 * math.pi * x)**4.0) + noise
```

We can test this function by first defining a grid-based sample of inputs from 0 to 1 with a step size of 0.01 across the domain.

```python
# grid-based sample of the domain [0,1]
X = np.arange(0, 1, 0.01)
```

We can then evaluate these samples using the target function without any noise to see what the real objective function looks like.

```python
# sample the domain without noise
y = [objective(x, 0) for x in X]
```

We can then evaluate these same points with noise to see what the objective function will look like when we are optimizing it.

```python
# sample the domain with noise
ynoise = [objective(x) for x in X]
```

We can look at all of the non-noisy objective function values to find the input that resulted in the best score and report it. This will be the optima, in this case, maxima, as we are maximizing the output of the objective function.

We would not know this in practice, but for out test problem, it is good to know the real best input and output of the function to see if the Bayesian Optimization algorithm can locate it.

```python
# find best result
ix = np.argmax(y)
print('Optima: x=%.3f, y=%.3f' % (X[ix], y[ix]))
```

    Optima: x=0.990, y=0.963

Finally, we can create a plot, first showing the noisy evaluation as a scatter plot with input on the x-axis and score on the y-axis, then a line plot of the scores without any noise.

```python
# plot the points with noise
pyplot.scatter(X, ynoise)
# plot the points without noise
pyplot.plot(X, y)
# show the plot
pyplot.show()
```

![png](https://dsm01pap004files.storage.live.com/y4mInhTedeUbNyjbXJ0sZaxsO299h_7sNc7c1hafPiWnRb85aje5EeKCiQCqgqHzHJfo8EgHJ2CDkKKFfWVk3yfptS_0fWYs8kDY-x_HEC3blYRG_Gk23Kd2lVfNEnuxpkMOf0Lw9ZAITS_G802vbv3GMPhNb3kJgubwFvsN872aGgaZJEfX9ZN5EEa0sQzNDgq?width=547&height=413&cropmode=none)

### Surrogate Function

The surrogate function is a technique used to best approximate the mapping of input examples to an output score.

Probabilistically, it summarizes the conditional probability of an objective function $(f)$, given the available data $(D)$ or $P(f|D)$.

A number of techniques can be used for this, although the most popular is to treat the problem as a regression predictive modeling problem with the data representing the input and the score representing the output to the model. This is often best modeled using a random forest or a Gaussian Process.

A Gaussian Process, or GP, is a model that constructs a joint probability distribution over the variables, assuming a multivariate Gaussian distribution. As such, it is capable of efficient and effective summarization of a large number of functions and smooth transition as more observations are made available to the model.

This smooth structure and smooth transition to new functions based on data are desirable properties as we sample the domain, and the multivariate Gaussian basis to the model means that an estimate from the model will be a mean of a distribution with a standard deviation; that will be helpful later in the acquisition function.

As such, using a GP regression model is often preferred.

We can fit a GP regression model using the GaussianProcessRegressor scikit-learn implementation from a sample of inputs $(X)$ and noisy evaluations from the objective function $(y)$.

First, the model must be defined. An important aspect in defining the GP model is the kernel. This controls the shape of the function at specific points based on distance measures between actual data observations. Many different kernel functions can be used, and some may offer better performance for specific datasets.

By default, a Radial Basis Function, or RBF, is used that can work well.

Once defined, the model can be fit on the training dataset directly by calling the fit() function.

The defined model can be fit again at any time with updated data concatenated to the existing data by another call to fit().

The model will estimate the cost for one or more samples provided to it.

The model is used by calling the predict() function. The result for a given sample will be a mean of the distribution at that point. We can also get the standard deviation of the distribution at that point in the function by specifying the argument return_std=True; for example:

This function can result in warnings if the distribution is thin at a given point we are interested in sampling.

Therefore, we can silence all of the warnings when making a prediction. The surrogate() function below takes the fit model and one or more samples and returns the mean and standard deviation estimated costs whilst not printing any warnings.

```python
from sklearn.gaussian_process import GaussianProcessRegressor
from scipy.stats import norm
from warnings import catch_warnings
from warnings import simplefilter
# surrogate or approximation for the objective function
def surrogate(model, X):
 # catch any warning generated when making a prediction
 with catch_warnings():
  # ignore generated warnings
  simplefilter("ignore")
  return model.predict(X, return_std=True)
```

We can call this function any time to estimate the cost of one or more samples, such as when we want to optimize the acquisition function in the next section.

For now, it is interesting to see what the surrogate function looks like across the domain after it is trained on a random sample.

We can achieve this by first fitting the GP model on a random sample of 100 data points and their real objective function values with noise. We can then plot a scatter plot of these points. Next, we can perform a grid-based sample across the input domain and estimate the cost at each point using the surrogate function and plot the result as a line.

We would expect the surrogate function to have a crude approximation of the true non-noisy objective function.

The plot() function below creates this plot, given the random data sample of the real noisy objective function and the fit model.

```python
# plot real observations vs surrogate function
def plot(X, y, model):
 # scatter plot of inputs and real objective function
 pyplot.scatter(X, y)
 # line plot of surrogate function across domain
 Xsamples = np.asarray(np.arange(0, 1, 0.001))
 Xsamples = Xsamples.reshape(len(Xsamples), 1)
 ysamples, _ = surrogate(model, Xsamples)
 pyplot.plot(Xsamples, ysamples)
 # show the plot
 pyplot.show()
```

```python
# sample the domain sparsely with noise
X = np.random.random(100)
y = np.asarray([objective(x) for x in X])
# reshape into rows and cols
X = X.reshape(len(X), 1)
y = y.reshape(len(y), 1)
# define the model
model = GaussianProcessRegressor()
# fit the model
model.fit(X, y)
# plot the surrogate function
plot(X, y, model)
```

![png](https://dsm01pap004files.storage.live.com/y4mQJ0vfVUOQqMZIicbxQqdhO_D6URlgeFjIwq0eE3jCC6AJOOpjHxIuSKUPPTtQHYu1tpMl_-OhzY4TBi2td5PlV5wj-2ElvHnYeBBiqhrzNtUKGaOHOYgckJboYhKNI-JXbfCDSpuUonqF94q9oOI1QeJyg72npjjzgYWAJcFGDYyySqOFunW3w8Hf8v-P8bT?width=559&height=413&cropmode=none)

Running the example first draws the random sample, evaluates it with the noisy objective function, then fits the GP model.

The data sample and a grid of points across the domain evaluated via the surrogate function are then plotted as dots and a line respectively.

Note: Your results may vary given the stochastic nature of the algorithm or evaluation procedure, or differences in numerical precision. Consider running the example a few times and compare the average outcome.

In this case, as we expected, the plot resembles a crude version of the underlying non-noisy objective function, importantly with a peak around 0.9 where we know the true maxima is located.

### Acquisition Function

The surrogate function is used to test a range of candidate samples in the domain.

From these results, one or more candidates can be selected and evaluated with the real, and in normal practice, computationally expensive cost function.

This involves two pieces: the search strategy used to navigate the domain in response to the surrogate function and the acquisition function that is used to interpret and score the response from the surrogate function.

A simple search strategy, such as a random sample or grid-based sample, can be used, although it is more common to use a local search strategy, such as the popular BFGS algorithm. In this case, we will use a random search or random sample of the domain in order to keep the example simple.

This involves first drawing a random sample of candidate samples from the domain, evaluating them with the acquisition function, then maximizing the acquisition function or choosing the candidate sample that gives the best score. The opt_acquisition() function below implements this.

```python
# optimize the acquisition function
def opt_acquisition(X, y, model):
 # random search, generate random samples
 Xsamples = np.random.random(100)
 Xsamples = Xsamples.reshape(len(Xsamples), 1)
 # calculate the acquisition function for each sample
 scores = acquisition(X, Xsamples, model)
 # locate the index of the largest scores
 ix = np.argmax(scores)
 return Xsamples[ix, 0]
```

The acquisition function is responsible for scoring or estimating the likelihood that a given candidate sample (input) is worth evaluating with the real objective function.

We could just use the surrogate score directly. Alternately, given that we have chosen a Gaussian Process model as the surrogate function, we can use the probabilistic information from this model in the acquisition function to calculate the probability that a given sample is worth evaluating.

There are many different types of probabilistic acquisition functions that can be used, each providing a different trade-off for how exploitative (greedy) and explorative they are.

Three common examples include:

Probability of Improvement (PI).
Expected Improvement (EI).
Lower Confidence Bound (LCB).
The Probability of Improvement method is the simplest, whereas the Expected Improvement method is the most commonly used.

In this case, we will use the simpler Probability of Improvement method, which is calculated as the normal cumulative probability of the normalized expected improvement, calculated as follows:

$$PI = cdf\left(\frac{\mu – best_\mu}{stdev}\right)$$
Where PI is the probability of improvement, cdf() is the normal cumulative distribution function, mu is the mean of the surrogate function for a given sample x, stdev is the standard deviation of the surrogate function for a given sample x, and best_mu is the mean of the surrogate function for the best sample found so far.

We can add a very small number to the standard deviation to avoid a divide by zero error.

The acquisition() function below implements this given the current training dataset of input samples, an array of new candidate samples, and the fit GP model.

```python
# probability of improvement acquisition function
def acquisition(X, Xsamples, model):
    # calculate the best surrogate score found so far
    yhat, _ = surrogate(model, X)
    best = max(yhat)
    # calculate mean and stdev via surrogate function
    mu, std = surrogate(model, Xsamples)
    mu = mu[0]
    # calculate the probability of improvement
    probs = norm.cdf((mu - best) / (std+1E-9))
    return probs
```

### Complete Bayesian Optimization Algorithm

We can tie all of this together into the Bayesian Optimization algorithm.

The main algorithm involves cycles of selecting candidate samples, evaluating them with the objective function, then updating the GP model.

```python
# perform the optimization process
for i in range(100):
    # select the next point to sample
    x = opt_acquisition(X, y, model)
    # sample the point
    actual = objective(x)
    # summarize the finding for our own reporting
    est, _ = surrogate(model, [[x]])
    print('>x=%.3f, f()=%3f, actual=%.3f' % (x, est, actual))
    # add the data to the dataset
    X = np.vstack((X, [[x]]))
    y = np.vstack((y, [[actual]]))
    # update the model
    model.fit(X, y)
```

    >x=0.730, f()=0.184289, actual=0.121
    >x=0.111, f()=-0.027583, actual=-0.074
    >x=0.552, f()=0.143937, actual=0.110
    >x=0.018, f()=-0.027990, actual=-0.092
    >x=0.183, f()=0.037242, actual=-0.161
    >x=0.314, f()=0.044528, actual=0.119
    >x=0.501, f()=0.092083, actual=0.225
    ......
    >x=0.031, f()=-0.041620, actual=0.012

```python
# example of bayesian optimization for a 1d function from scratch
from math import sin
from math import pi
from numpy import arange
from numpy import vstack
from numpy import argmax
from numpy import asarray
from numpy.random import normal
from numpy.random import random
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from warnings import catch_warnings
from warnings import simplefilter
from matplotlib import pyplot

# objective function
def objective(x, noise=0.1):
    noise = normal(loc=0, scale=noise)
    return (x**2 * sin(5 * pi * x)**6.0) + noise

# surrogate or approximation for the objective function
def surrogate(model, X):
    # catch any warning generated when making a prediction
    with catch_warnings():
        # ignore generated warnings
        simplefilter("ignore")
        return model.predict(X, return_std=True)

# probability of improvement acquisition function
def acquisition(X, Xsamples, model):
    # calculate the best surrogate score found so far
    yhat, _ = surrogate(model, X)
    best = max(yhat)
    # calculate mean and stdev via surrogate function
    mu, std = surrogate(model, Xsamples)
    mu = mu[0]
    # calculate the probability of improvement
    probs = norm.cdf((mu - best) / (std+1E-9))
    return probs

# optimize the acquisition function
def opt_acquisition(X, y, model):
    # random search, generate random samples
    Xsamples = random(100)
    Xsamples = Xsamples.reshape(len(Xsamples), 1)
    # calculate the acquisition function for each sample
    scores = acquisition(X, Xsamples, model)
    # locate the index of the largest scores
    ix = argmax(scores)
    return Xsamples[ix, 0]

# plot real observations vs surrogate function
def plot(X, y, model):
    # scatter plot of inputs and real objective function
    pyplot.scatter(X, y)
    # line plot of surrogate function across domain
    Xsamples = asarray(arange(0, 1, 0.001))
    Xsamples = Xsamples.reshape(len(Xsamples), 1)
    ysamples, _ = surrogate(model, Xsamples)
    pyplot.plot(Xsamples, ysamples)
    # show the plot
    pyplot.show()

# sample the domain sparsely with noise
X = random(100)
y = asarray([objective(x) for x in X])
# reshape into rows and cols
X = X.reshape(len(X), 1)
y = y.reshape(len(y), 1)
# define the model
model = GaussianProcessRegressor()
# fit the model
model.fit(X, y)
# plot before hand
plot(X, y, model)
# perform the optimization process
for i in range(100):
    # select the next point to sample
    x = opt_acquisition(X, y, model)
    # sample the point
    actual = objective(x)
    # summarize the finding
    est, _ = surrogate(model, [[x]])
    print('>x=%.3f, f()=%3f, actual=%.3f' % (x, est, actual))
    # add the data to the dataset
    X = vstack((X, [[x]]))
    y = vstack((y, [[actual]]))
    # update the model
    model.fit(X, y)

# plot all samples and the final surrogate function
plot(X, y, model)
# best result
ix = argmax(y)
print('Best Result: x=%.3f, y=%.3f' % (X[ix], y[ix]))
```

![png](https://dsm01pap004files.storage.live.com/y4m8ahGYviDQ901UeEgThVseAhF9Xlo7M9Z5n41GdPAkRzcNkkkwa-2UtuTIeLK3179sj7Pqn8FdNF1QV6_o0MLClWzRnfDa5gG9Z0NAS4LRAgyE_7gMkvWlHeMtRnEFWIRxr0gYav0rZRc5ri9UxpiXercaYGth6SmLnHnVjiQaVigPh6cRenr6cpW5ScURLGg?width=559&height=413&cropmode=none)

    >x=0.102, f()=0.060782, actual=0.292
    >x=0.873, f()=0.413320, actual=0.536
    >x=0.843, f()=0.320122, actual=-0.048
    >x=0.891, f()=0.448285, actual=0.706
    >x=0.616, f()=0.048923, actual=-0.051
    >x=0.908, f()=0.495629, actual=0.960
    ......
    >x=0.842, f()=0.298019, actual=0.010
    >x=0.032, f()=0.029685, actual=0.024
    >x=0.167, f()=0.058235, actual=0.048
    >x=0.558, f()=0.125632, actual=0.080

![png](https://dsm01pap004files.storage.live.com/y4ma1jfCa3KlXDBUj70DlKbLU0XsflExx5ueX0qX5wM4ZmwRqOWfyCMrS0c8G57gOvD5ofEwFOWw6XQaef10dgaPL_CA8h57iTz4ZXjk2APmSCakbD1bbbT9VBD9wKJd79wkSYBqBYp1er2qht8LmjfwH6ppn8GXJIBfimPdF3KRWF1kxnXpKW0_aLoGBk6xhf2?width=559&height=413&cropmode=none)

    Best Result: x=0.908, y=0.960

## Hyperparameter Tuning With Bayesian Optimization

It can be a useful exercise to implement Bayesian Optimization to learn how it works.

In practice, when using Bayesian Optimization on a project, it is a good idea to use a standard implementation provided in an open-source library. This is to both avoid bugs and to leverage a wider range of configuration options and speed improvements.

Two popular libraries for Bayesian Optimization include Scikit-Optimize and HyperOpt. In machine learning, these libraries are often used to tune the hyperparameters of algorithms.

Hyperparameter tuning is a good fit for Bayesian Optimization because the evaluation function is computationally expensive (e.g. training models for each set of hyperparameters) and noisy (e.g. noise in training data and stochastic learning algorithms).

In this section, we will take a brief look at how to use the Scikit-Optimize library to optimize the hyperparameters of a k-nearest neighbor classifier for a simple test classification problem. This will provide a useful template that you can use on your own projects.

The Scikit-Optimize project is designed to provide access to Bayesian Optimization for applications that use SciPy and NumPy, or applications that use scikit-learn machine learning algorithms.

```python
!pip install scikit-optimize
```

    Looking in indexes: https://pypi.tuna.tsinghua.edu.cn/simple
    Requirement already satisfied: scikit-optimize in /Users/junxiaoguo/opt/anaconda3/envs/ml/lib/python3.10/site-packages (0.9.0)
    Requirement already satisfied: joblib>=0.11 in /Users/junxiaoguo/opt/anaconda3/envs/ml/lib/python3.10/site-packages (from scikit-optimize) (1.1.1)
    Requirement already satisfied: numpy>=1.13.3 in /Users/junxiaoguo/opt/anaconda3/envs/ml/lib/python3.10/site-packages (from scikit-optimize) (1.23.5)
    Requirement already satisfied: pyaml>=16.9 in /Users/junxiaoguo/opt/anaconda3/envs/ml/lib/python3.10/site-packages (from scikit-optimize) (21.10.1)
    Requirement already satisfied: scikit-learn>=0.20.0 in /Users/junxiaoguo/opt/anaconda3/envs/ml/lib/python3.10/site-packages (from scikit-optimize) (1.2.1)
    Requirement already satisfied: scipy>=0.19.1 in /Users/junxiaoguo/opt/anaconda3/envs/ml/lib/python3.10/site-packages (from scikit-optimize) (1.10.0)
    Requirement already satisfied: PyYAML in /Users/junxiaoguo/opt/anaconda3/envs/ml/lib/python3.10/site-packages (from pyaml>=16.9->scikit-optimize) (6.0)
    Requirement already satisfied: threadpoolctl>=2.0.0 in /Users/junxiaoguo/opt/anaconda3/envs/ml/lib/python3.10/site-packages (from scikit-learn>=0.20.0->scikit-optimize) (2.2.0)

```python
# example of bayesian optimization with scikit-optimize
from numpy import mean
from sklearn.datasets import make_blobs
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from skopt.space import Integer
from skopt.utils import use_named_args
from skopt import gp_minimize

# generate 2d classification dataset
X, y = make_blobs(n_samples=500, centers=3, n_features=2)
# define the model
model = KNeighborsClassifier()
# define the space of hyperparameters to search
search_space = [Integer(1, 5, name='n_neighbors'), Integer(1, 2, name='p')]

# define the function used to evaluate a given configuration
@use_named_args(search_space)
def evaluate_model(**params):
 # something
 model.set_params(**params)
 # calculate 5-fold cross validation
 result = cross_val_score(model, X, y, cv=5, n_jobs=-1, scoring='accuracy')
 # calculate the mean of the scores
 estimate = mean(result)
 return 1.0 - estimate

# perform optimization
result = gp_minimize(evaluate_model, search_space)
# summarizing finding:
print('Best Accuracy: %.3f' % (1.0 - result.fun))
print('Best Parameters: n_neighbors=%d, p=%d' % (result.x[0], result.x[1]))
```

    Best Accuracy: 0.976
    Best Parameters: n_neighbors=5, p=2

Running the example executes the hyperparameter tuning using Bayesian Optimization.

```python

```
