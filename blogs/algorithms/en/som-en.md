# Machine learning --- Self-organizing map network SOM clustering algorithm

SOM (Self-Origanizing Maps), or SOFM (Self-Origanizing Feature Maps), is a clustering algorithm based on neural networks, which is an unsupervised learning algorithm.

## The basic structure of SOM

SOM is a single-layer neural network, that is to say ---- **SOM algorithm only contains input layer and computational layer (Computational Layer)**

<p align='center'>
<img src='https://bbs-img.huaweicloud.com/blogs/img/som-1.png' width=450>
<br>
  Structure of SOM
</p>

The calculation layer of the SOM is also called the competition layer or the output layer. This layer is a structure composed of a series of neurons (neurons). (In a two-dimensional structure), where the nodes of the computing layer and the nodes of the input layer are fully connected. The computing layer can discretize the input of any dimension into a one-dimensional or two-dimensional discrete space, that is, The SOM algorithm can play a role in dimensionality reduction-mapping high-dimensional data input to the dimensional space defined by the computing layer. In k-means, we need to specify the value of k in advance, that is, the number of clusters But in SOM, you need to specify a topological structure (the geometric relationship between clusters) in advance. In the topology, a node is a cluster.

1.The goal of SOM is to use points in a low-dimensional (usually two-dimensional or three-dimensional) target space to represent all points in a high-dimensional space, and to maintain the distance and proximity relationship (topological relationship) between the points as much as possible.
2. The SOM algorithm is different from other artificial neural network (ANN) models. SOM uses **Competitive Learning** instead of error correction learning (such as backpropagation neural networks). And SOM uses the neighborhood function to Keep the topological properties of the input

## SOM learning process

### 1. Competitive Process

The physiological basis of competitive learning rules is the phenomenon of lateral inhibition of nerve cells: when a nerve cell is excited, it will have an inhibitory effect on the surrounding nerve cells. The strongest inhibitory effect is the "I alone" of competition to win. This approach is called Winner Take All (Winner Take All). Competitive learning rules are obtained from the phenomenon of side inhibition of nerve cells.

In SOM, each neuron in the competition layer will refer to the input N-dimensional data (x) and the connection weight (weight w). The range of w is between {0,1}, with any value of normalized (Normalized) To initialize. During the learning process, the distance between the input x and the weight w of all neurons is calculated. When the distance is the smallest, the neuron becomes the winner, which is the process of competition.

### 2. Cooperation Process

This process allows the winner of the competition process and its neighboring neurons to learn from the input data provided. In order to more sensitively form a map in the competitive hierarchy for similar features, the "winner" neuron is based on a fixed function. Determine the neighboring neuron, and at the same time the corresponding weight of this neuron will be updated. The philosophy of Kohonen network is "Winner Take All", that is, only the winner has the output, and only the winner can update the weight w .

### 3. Adaptation Process

This process adapts the activation function to make the winner neuron and neighboring neuron more sensitive to specific input values, while also updating the corresponding weights. Through this process, neurons that are adjacent to the winner neuron will adapt more than those that are far away. The size of the adaptation is controlled by the learning rate. The learning rate decays with the learning time, which reduces the convergence speed of the SOM.

## SOM algorithm flow

### 1. Vector normalization

Normalize the input vector in the SOM network and the weight vector corresponding to the neuron in the competition layer to obtain$\hat{X},\hat{W}$

$$\hat{X}={X\over{||X||}}, \hat{W}={W\over{||W||}}$$

### 2. Random sampling, and find the winning neuron (best matching unit, BMU)

First, randomly take a sample from $X$

Use the Euclidean distance formula to calculate the similarity between the input vector and the weight vector of the nodes in the competition layer. $\hat{W_j}(j=1,2,3,4,5...m)$ Perform similarity comparison, the most similar neuron wins, and the weight vector is $\hat{w_j}$:

$$i(x)=\argmin_j{||x(n)-w_j(n)||}$$

### 3. Update the weights of the winning neuron and neighboring neurons

Update the weight vector of the nodes near the BMU (including the BMU itself) to the input vector

$$W_v(s+1)=W_v(s)+\theta(u,v,s)\cdot\alpha(s)\cdot(D(t)-W_v(s))$$

in,
$s$ represents the current number of iterations
$\theta$ represents the constraint caused by the distance from the BMU, usually called the neighborhood function
$\alpha$ represents the learning rate
$D$ represents the input vector

### 4. Repeat from step 2 until the number of iterations reaches the upper limit

<p align='center'>
<img src='https://bbs-img.huaweicloud.com/blogs/img/som-2.gif' width=300>
<br>SOM Learning Procedure
</p>

## Advantages and disadvantages

### advantage

-SOM imposes the neighbor relationship on the cluster centroid, which is conducive to the interpretation of the clustering results
-With dimensionality reduction function
-Strong visualization
-Self-organization ability
-Unsupervised learning
-Keep topology information

### shortcoming

-The main disadvantage of SOM is that the algorithm requires neuron weights to be necessary and sufficient to cluster the input. If SOM provides too little or too much information in the weight layer, it will have a greater impact on the result
-Need to define domain function
-SOM lacks a specific objective function
-A SOM cluster usually does not correspond to a single natural thickness, and there may be merging and splitting of natural clusters
-SOM does not guarantee convergence
