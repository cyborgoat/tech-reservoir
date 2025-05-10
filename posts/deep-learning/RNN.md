---
title: "Understanding RNN architecture"
date: "2022-08-05"
author: "Junxiao Guo"
tags: ["deep-learning", "NLP"]
excerpt: "An introduction to recurrent neural network (RNN) by using several examples."
---


In recent years, natural language processing has become a hot topic in the field of artificial intelligence. Model structures such as LSTM, Attention, and Transformer have become popular. Based on these theories, various powerful pre-training models have been derived, such as BERT, GPT3, etc.

One of the common core ideas of these algorithms is RNN (Recurrent Neural Network). This article will introduce the logic and implementation principles of RNN in as much detail as possible (the core formula will be included, and the formula derivation of the specific training process will not be introduced)

## Limitations of Traditional Neural Networks

In neural network algorithms, most of the algorithms (MLP, CNN,...) are independently corresponding to the input (x) and the output (y), that is to say

$$x_1 \rightarrow y_1 , x_2 \rightarrow y_2 , ... , x_n \rightarrow y_n$$

But in some scenarios, independent input becomes insufficient, for example, we want to fill in an incomplete sentence

<center>The kind of meat we need to cook steak is __</center>

Obviously, the answer here should be "beef". But in the case of using neural networks, even if we do word segmentation on the above sentence, it is obviously impossible to predict the result based on a certain word or word. At this time, we need To process **interdependent time series data**, in this scenario, it needs to be implemented based on other methods.

## The Sliding Predictor

### Sliding Predictor & CNN

In order to solve the above problems, one of the most concise and easy-to-understand methods is the sliding prediction model (Sliding Predictor), which takes the input of the previous few time nodes and the input of the current time node as the overall input of the model. The following is for more clarity The display model structure, we take a more classic Use Case as an example: predicting stock prices

![]()
<p align='center'>
    <img src="https://bbs-img.huaweicloud.com/blogs/img/sliding-predictor.PNG">
    <center>Sliding Predictor Example</center>
</p>

As shown above, the Sliding Predictor will use the stock vector (stock vector) at $t, t+1, t+2, t+3$ as the overall input to predict the stock price at $t+3$

<p align='center'>
    <img src="https://bbs-img.huaweicloud.com/blogs/img/sliding-predictor-2.PNG">
    <center>Sliding Predictor Example</center>
</p>

By analogy, Sliding Predictor will use the same logic to predict the stock price at $t+4$

It is not difficult to see that the above calculation method is very similar to the convolution process (Convolution Step) in the computer vision algorithm CNN, so the Sliding Predictor is actually a CNN applied to sequence data. Such an algorithm is also called *Time-Delay neural network*

### Finite-response Model

Such a model belongs to the *Finite-response Model*. To put it more vividly, what happens today will only affect the results within $N$ days in the future, where $N$ is the width of the entire system

$$Y_t=f(X_t,X_{t-1},...,X_{t-N})$$

### Problems

The above model looks very reasonable, but what if our influence radiation width becomes larger? What if what happens today will affect the results in the next 10,000 days? At this time, the model will become more complicated

> "Don't worry, our CPU is enough" --> Do we?

## Long Term Dependency

In many scenarios, the results we need to predict will be based on long term dependencies. For example, in stock forecasting, we may consider:

- stock market trend within a week
- Stock market trend within a month
- Stock market trends throughout the year
-...

### NARX Network

What if what happens today affects all future outcomes? We need infinite memory:

<p align='center'>
    <img src="https://bbs-img.huaweicloud.com/blogs/img/infinite-response.PNG">
  <center>Infinite-response Model</center>
</p>

Then,

$$Y_t=f(x_t,x_{t-1},...,x_{t-\infty})$$

Or we can think as,

$$Y_t=f(X_t,Y_{t-1})$$

- Such an assumption needs to define the initial state, which is $Y_{-1}$ corresponding to $t=0$,
- At this time, the input $X_0$ corresponding to $t=0$ will synthesize $Y_{-1}$ to get $Y_0$,
- Then get $Y_1$, $Y_2$,...,$Y_\infty$ through $Y_0$, even when $X_1,...,X_{\infty}$ are 0
  - i.e. input without $X$ at the corresponding moment

Such a model structure is called *NARX network (nonlinear autoregressive network with exogenous inputs)*

$$Y_t=f(X_{0:t},Y_{0:t-1})$$

<p align='center'>
    <img src="https://bbs-img.huaweicloud.com/blogs/img/narx.PNG">
  <center>NARX Model Structure</center>

</p>

A more general NARX is as follows:

<p align='center'>
    <img src="https://bbs-img.huaweicloud.com/blogs/img/narx-2.PNG">
    <center>NARX Model Structure</center>
</p>

$Y_t$ at time $t$ will be calculated based on previous K outputs $Y_{t-1},...,Y_{t-k}$ and L inputs $X_t,...,X_{t-L}$

The "full" NARX is as follows:

<p align='center'>
    <img src="https://bbs-img.huaweicloud.com/blogs/img/narx-3.PNG">
    <center>NARX "full" Model Structure</center>
</p>

$Y_t$ at time $t$ will be calculated based on all previous outputs $Y_{t-1},...,Y_{-1}$ and all inputs $X_t,...,X_{0}$, This model cannot be done due to computing power because it is a practical model.

### System definition memory

In the temporal model, the definition of explicit memory is to let him remember:

$$m_t=r(y_{t-1},h_{t-1},m_{t-1})$$
$$h_t=f(x_t,m_t)$$
$$y_t=g(h_t)$$

Among them, $m_t$ represents the "memory" variable, which is used to "remember" the past, and is generally stored in the "memory unit" of the model

### Jordan Network

In 1986, M.I.Jordan's paper "Serial order: A parallel distributed processing approach", defined the memory unit as the average of all historical outputs (running average)

<p align='center'>
   <img src="https://bbs-img.huaweicloud.com/blogs/img/elman-1.PNG">
  <center>Elman Network</center>
</p>

## The State-space Model & RNN

### State-space Model & Single Hidden Layer RNN

Another model for infinite response system: the state-sapce model

$$h_t=f(x_t,h_{t-1})$$
$$y_t=g(t_t)$$

- The state (State) of the neural network represented by $h_t$ integrates all historical information
  - The model will directly embed memory into this layer
- The model needs to define the initial state $h_{-1}$
- This is a complete RNN model

The simplest state-space model structure is as follows,

<p align='center'>
    <img src="https://bbs-img.huaweicloud.com/blogs/img/state-space-1.PNG">
  <center>The simple state-space model</center>
</p>

- The green block represents the state defined based on the input and the preorder state at any time
- Input at time $t=0$ will permanently affect subsequent output
- This model is actually the standard `Single Hidden Layer RNN`

### Multiple recurrent layer RNN

The multi-layer cycle RNN structure is as follows

<p align='center'>
    <img src="https://bbs-img.huaweicloud.com/blogs/img/rnn-1.PNG">
    <center>Multiple recurrent layer RNN</center>
</p>

or,

<p align='center'>
    <img src="https://bbs-img.huaweicloud.com/blogs/img/rnn-2.PNG">
    <center>Multiple recurrent layer RNN</center>
</p>

even can be,

<p align='center'>
    <img src="https://bbs-img.huaweicloud.com/blogs/img/rnn-3.PNG">
    <center>Multiple recurrent layer RNN</center>
</p>

It is also possible to generalize based on other recursions,

<p align='center'>
    <img src="https://bbs-img.huaweicloud.com/blogs/img/rnn-4.PNG">
    <br>
    <center>Generalization with other recurrences</center>
</p>

## Core Formula

For the following structure,

<p align='center'>
    <img src="https://bbs-img.huaweicloud.com/blogs/img/rnn-5.PNG">
    <br>
</p>

$$h_i^{1}(-1) = part \quad of \quad network \quad  parameters$$
$$h_i^{1}(t) = f_1(\sum_j{w_{ji}^{1}{X_j(t)}+\sum{w_{ji}^{1}h_i^{(1)}(t-1)+b_i^{(1)}}})$$
$$Y(t)=f_2(\sum{w_{ji}^{2}h_i^{(1)}(t-1)+b_i^{(2)}},k=1..M)$$

## Variant of RNN

The RNN model usually has two variants,
<p align='center'>
    <img src="https://bbs-img.huaweicloud.com/blogs/img/rnn-6.PNG">
    <br>
</p>

The specific application is as follows,

1. Delayed Sequence to Sequence, such as Machine Translation
2. Sequence to Sequence, such as stock prediction, label prediction, etc.

## Summarize

This article mainly summarizes the origin of RNN and the overall structural details. The specific training steps (such as back propagation, etc.) are not explained here due to the cumbersome formulas. Interested friends can go down and learn about it. The overall gradient The descent method is not much different from the traditional neural network.