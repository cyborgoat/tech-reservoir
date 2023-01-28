---
title: Apriori & FP-growth Algorithm
summary: A brief summary of commands to install postgres database
author: Junxiao Guo
date: 2021-01-01
tags:
  - machine-learning
  - data-mining
---

When you buy vegetables, have you ever made a list of items to buy? Everyone has different needs and preferences when making the list. As the store itself, it can be better according to the category of the item and the frequency of purchase To understand the customer's consumption habits. Assuming that many customers like their colleagues to buy X and Y two things, then:

1. X and Y products can be placed on the same product shelf, so consumers can purchase these two items more conveniently
2. Coupons that can be purchased at the same time can be provided on X and Y products to stimulate consumption
3. Targeted advertisements about Y products for customers who often buy X products

Now that you have seen the value that can be generated from frequent item information mining, how do we achieve it?

## Association rules Analysis

Association rule analysis can be used to explain how two items are related. There are currently three popular ways to measure the degree of association.

### 1. Support

The degree of support expresses the "popularity" of the item. Using the following illustration, the degree of support for apples is the probability of apples appearing in 8 transactions.

![apriori-img-1](https://bbs-img.huaweicloud.com/blogs/img/0.png)

### 2. Confidence

Confidence refers to the probability that event B will appear after event X appears. The following legend means that it is known that apples have been purchased, and the amount of beer purchase confidence formula

<center><img src='https://bbs-img.huaweicloud.com/blogs/img/2(30).png' width=300></center>

Note that the shortcomings of the confidence formula are more obvious. For example, beer itself is very popular, so the confidence in buying beer after buying apples will also be high. So it may be misleading, and the third way is fine to olve such a problem.

### 3. Lift

The lift indicates the possibility of Y appearing after the event X appears and the popularity of the event Y is known. The following figure explains the calculation method of the lift

<center><img src='https://bbs-img.huaweicloud.com/blogs/img/3(27).png' width=300></center>

## Apriori algorithm

The purpose of Apriori's algorithm is to reduce the number of events that need to be calculated. The principle of the Aprioi algorithm is: If event X is infrequent, then all of his supersets are infrequent. More specifically, assume that beer is an infrequent For products that are frequently purchased, {beer, fried chicken} is the same or more infrequent. So when we are doing frequent item mining, we no longer consider {beer, fried chicken} or any combination that contains beer. .

So, what is called high frequency and what is called low frequency? The specific algorithm flow of Apriori will explain your problem.

1. Start with a single sample, such as {beer}, {fried chicken} and {beef}
2. Calculate support for each single product, and keep all items higher than the minimum support threshold**
3. Keep all the items in step 2 and find the pairwise combination of all the remaining items
4. Repeat step2-3, note that in step3, the number of combinations increases by one each time, for example, the first time is {beer, fried chicken}, then the next cycle is {beer, fried chicken, beef}

The above-mentioned **minimum support threshold** is mainly determined by expert experience. At the same time, in stpe2, confidence or lift can be used. Which parameter to use depends on the distribution of the data itself

### Limitations of Apriori

-High computational complexity: Although the Apriori algorithm reduces the number of events that need to be calculated, it still has a large number of calculations when the amount of events is particularly large.
-Complex event combination methods: When analyzing very complex item combinations, we need to lower the minimum support threshold, otherwise it may not be possible to extract the relevant combinations. However, at the same time, there will also be a large number of complex combinations Interference analysis occurs.

## FP-growth algorithm

In order to solve the limitation of Apriori, the FP-growth algorithm is based on the Apriori principle and stores the data set in the FP (Frequent Pattern) tree to find frequent itemsets. The FP-growth algorithm only needs to scan the database twice, while the Apriori algorithm needs to scan the data set once to find each potential frequent item set. The process of the algorithm to find the frequent item set is:

1. Construct FP tree;

2. Mining frequent itemsets from the FP tree.

### FP tree

FP represents a frequent pattern, which connects similar elements through links, and the connected elements can be regarded as a linked list. After sorting the data items corresponding to each transaction in the transaction data table according to the support degree, insert the data items in each transaction in descending order into a tree with NULL as the root node, and record at each node at the same time The degree of support the node appears.
