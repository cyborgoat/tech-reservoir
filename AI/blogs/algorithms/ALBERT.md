# 自然语言处理 --- ALBERT

[\[论文链接\]](https://arxiv.org/pdf/1909.11942.pdf)

> 本文末会提供一个基于Transformers(深度学习开源库)的简易ALBERT算法多选题任务推理Demo(暂不提供Fine Tuning代码)

ALBERT: A LITE BERT FOR SELF-SUPERVISED LEARNING OF LANGUAGE REPRESENTATIONS

在NLP模型预训练过程中, 提升模型规模往往可以提高模型表现. 但是, 很多时候由于GPU内存有限的原因, 导致了模型训练时间变长. ALBERT提出了两种参数压缩的方法来降低BERT的内存占用,同时提高训练速度:

- 第一种方法是通过嵌入参数因式分解(Factorized Embedding Parameterization),将巨大的词嵌入矩阵分解为两个小型矩阵. 
- 第二种方法是跨层参数共享(Cross-Layer Parameter Sharing).

通过以上两种方法, 一个近似于BERT-large参数配置的ALBERT模型只需要原模型$\frac{1}{18}$的参数量就能达到1.7倍的训练速度.

## ALBERT核心结构

### 模型结构选择

ALBERT的基础模型与BERT非常类似(Transformer Encoder + GELU nonlinearities). 首先,我们做出以下定义(以下定义和BERT原文中的参数定义进行了对齐):

- E: Embedding size
- L: Encoder layers
- H: Hidden size

其中, feed-forward/filter尺寸为 $4H$, attenion heads为 $H/64$

### 嵌入参数分解(Factorized embedding parameterization)

在BERT, XLNet, RoBERTa 模型中, WordPiece embedding size E 与隐藏层是完全对齐的, i.e.$E \equiv H$, 但是这样的选择对模型训练和实用性来说并非最优, 原因如下

- 从模型角度来讲, WordPiece embeeding的意义在于学习到上下文独立(context-independent)的表征, hidden-layer embeddings 的意义在于学习到依赖于上下文(context-dependent)的表征, 类似于BERT结构的模型的强劲表现主要来源于上下文表征的获取. 因此, 将WordPiece embedding size 和hidden-layer size 解耦可以更使模型更有效的利用好参数, 也就是说, $H \gg E$

- 从实用性的角度来讲, NLP任务一般会要求vocabulary size $V$ 非常大, 如果 $E\equiv H$的话, 提高$H$也会提高嵌入矩阵的大小,也就是$V×E$. 这样会导致出现数十亿级别的参数, 但是大部分在训练过程中都是稀疏的.

因此, ALBERT将嵌入参数(embedding parameters)进行了矩阵分解, 分解为了两个相对小很多的矩阵. 不像直接将one-hot 向量直接放入巨大的的隐藏层($size=H$), 而是将这些向量先映射到一个更低维度的矩阵空间($size=E$)中, 再映射到隐藏层中. 通过矩阵分解, ALBERT将嵌入参数的矩阵大小由$O(V \times H)$变成了$O(V \times E + E \times H)$. 这种转换可以在当$H \gg E$的时候大幅度降低参数量.  

### 跨层参数共享(Cross-layer parameter sharing)

参数共享的方式有很多,比如:

- 只共享正向传播(feed-forward network FFN)的参数
- 只共享注意力(attention)参数
- ...

ALBERT的默认方式为共享所有层的所有参数.

### 句间连贯性损失(Inter-sentence coherence loss)

除了masked language modeling (MLM) loss以外, BERT还使用了 next-sentence prediction (NSP) loss. NSP loss是一个用于预测两个片段是否是在原句中连贯的损失函数: 正样本通过抽取原文本中的连续片段来获得; 负样本通过组合不同文本中的变短来生成; 其中正负样本的比例相等. NSP的目标是为了改善下游任务的能力, 例如一些需要进行两个句子之间的自然语言推理的任务. 但是已有部分研究发现NSP的设计不稳定可靠,所以后来又移除了NSP. ALBERT一文推测NSP不具有有效性的原因是该方法在单个任务中混淆了**主题预测(topic predicion)**和**连贯性预测(coherence prediction)**.

基于以上的分析,ALBERT从着重于连贯性信息的角度提出了**sentence-order predicion(SOP) loss**, SOP的正样本使用了和BERT完全相同的采样方法, 但是负样本变成了**正样本的逆序样本**, 通过这样的方式可以强制模型学习句间的连贯性特征.

### 模型配置

Table 1. 展示了BERT和ALBERT的参数间区别. 可以看到, ALBERT-xlarge的参数量仅约等于BERT-base的60%左右

<p>
    <img src='https://bbs-img.huaweicloud.com/blogs/img/20211130/1638243561546013186.PNG'>
    <center>Table 1: BERT 和 ALBERT的参数配置</center>
</p>

## ALBERT多选题任务推理demo

```python
import torch
from transformers import AlbertTokenizer, AlbertForMultipleChoice
from transformers import BertTokenizer, AlbertModel
```


```python
tokenizer = AlbertTokenizer.from_pretrained('albert_chinese_small')
model = AlbertForMultipleChoice.from_pretrained('albert_chinese_small')
```

    The tokenizer class you load from this checkpoint is not the same type as the class this function is called from. It may result in unexpected tokenization. 
    The tokenizer class you load from this checkpoint is 'AlbertTokenizer'. 
    The class this function is called from is 'BertTokenizer'.
    Some weights of the model checkpoint at pre_trained_models/albert_chinese_small were not used when initializing AlbertForMultipleChoice: ['predictions.LayerNorm.bias', 'predictions.dense.weight', 'predictions.dense.bias', 'predictions.bias', 'predictions.LayerNorm.weight', 'predictions.decoder.weight', 'predictions.decoder.bias']
    - This IS expected if you are initializing AlbertForMultipleChoice from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
    - This IS NOT expected if you are initializing AlbertForMultipleChoice from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
    Some weights of AlbertForMultipleChoice were not initialized from the model checkpoint at pre_trained_models/albert_chinese_small and are newly initialized: ['classifier.bias', 'classifier.weight']
    You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
    


```python
# 话题
prompt = "算法组的成员们精通各种AI算法." 

# 选择
choice0 = "成都动物园里的大熊猫人人都想摸一下."
choice1 = "他们主要负责AI算法的研究和落地."
choice2 = "我没吃上吉士果."
choice3 = "你吃饭了吗?"
choices = [choice0,choice1,choice2,choice3]
labels_list = [0,1,0,0]
labels = torch.FloatTensor(labels_list).unsqueeze(0)  # choice0 is correct, , batch size 1
size = len(choices)
```


```python
encoding = tokenizer([prompt for _ in range(size)],choices, return_tensors='pt', padding=True)
outputs = model(**{k: v.unsqueeze(0) for k, v in encoding.items()}, labels=labels)  # batch size is 1
loss = outputs.loss
logits = outputs.logits
print(logits)
```

    tensor([[ 0.0100,  0.2225, -0.0217, -0.0517]], grad_fn=<ViewBackward0>)
    


```python
from torch import nn
from torch import autograd
m = nn.Softmax(dim=1)
soft_maxed_logits = m(logits)
print(soft_maxed_logits)
```

    tensor([[0.2412, 0.2983, 0.2337, 0.2268]], grad_fn=<SoftmaxBackward0>)
    


```python
result = torch.argmax(soft_maxed_logits).detach().numpy()
result
```




    array(1, dtype=int64)




```python
# 反馈最终选择
choices[result]
```




    '他们主要负责AI算法的研究和落地.'


