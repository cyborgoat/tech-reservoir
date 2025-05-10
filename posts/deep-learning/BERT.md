---
title: "Pre-trained Model --- BERT"
date: "2021-03-05"
author: "Junxiao Guo"
tags: [""pre-trained model"", "nlp"]
excerpt: "An introduction to BERT algorithm and a simple example to use this model"
---


## Abstract

BERT 全称为 Bidirectional Encoder Representations from Transformers. BERT 旨在通过对所有层的左右上下文进行联合调节,从未标记的文本中预训练深度双向表示. 因此,预训练的 BERT 模型可以仅通过一个额外的输出层进行微调,从而为各种任务(例如问答和语言推理)创建最先进的模型,而无需对特定于任务的架构进行大量修改.

## Masked Language Model

MLM (Masked Language Model) 灵感来源于完形填空 (Taylor, 1953): 从句子中删除一个或几个单词并要求学生填写缺失内容的活动. 该句子可以称为"stem", 删除的术语本身称为"key". MLM将输入的tokens进行随机掩盖, 然后根据上下文信息训练模型来预测被掩盖的tokens. MLM与通常的 left-to-right 模型不同, MLM 目标使表示能够融合左右上下文, 这使其能够预训练深度双向 Transformer. 除了Masked Language Model,BERT还使用了 "Nets sentence prediction" 任务来联合预训练 text-pair representations.

## BERT

BERT的整体流程包含两个步骤: `pre-training` and `fine-tuning`.

在 pre-training 中, 模型会在无监督的情况下通过不同的预训练任务中进行训练. 在 fine-tuning 过程中, BERT 首先根据 pre-training 过程中学习到的参数进行初始化, 所有的参数会通过有监督的方式来训练下游任务. 每个下游任务都有单独的微调模型, 即使它们是使用相同的预训练参数初始化的.

<img src='https://bbs-img.huaweicloud.com/blogs/img/paperfig-1.PNG'>
BERT的 pre-training 和 fine-truning的整体流程

BERT的特点之一就是它在不同任务下的的统一化结构, 这使得能够最小化预训练过程和微调过程中的模型结构.

**Model Architecture**

BERT的的模型结构是 Multi-layer Bidirectional Transformer Encoder (多层双向Transformer编码器). 其中 `L` 表示模型的层数 (i.e., Transformer blocks), `A` 表示self-attention heads.

BERT主要提供了两个模型的结果:

1. $BERT_{BASE}$ (L=12,H=768,A=12) 总计参数量为110M.
2. $BERT_{LARGE}$ (L=24,H=1024,A=16) 总计参数量为340M.

$BERT_{BASE}$的结构与*OpenAI GPT*的结构完全相等, 该设计主要是为了和*OpenAI GPT*进行比较, 其中值得注意的是: **BERT Transformer使用的是bidirectional self-attention, GTP Transformer 使用的是constrained self-attention, 所有的token只注意上文的信息,并不包含下文信息**

**I/O Representations**

为了让 BERT 处理各种下游任务,模型的输入可以明确地表示单个句子和句子对(Sentence Pair)(e.g.,  <Question,Answer>), 并将其转化为单个 token sequence. 在整个工作中,"句子"可以是任意范围的连续文本, 而不是实际的语言句子. "序列" 是指输入到 BERT 的 token sequence, 可以是单个句子, 也可以是两个打包在一起的句子.

BERT 使用了 *WordPiece embeddings* 做为参考, 其中包含了30,000个 token. 对于所有的序列来说, 首个 token 永远都是一个特殊的分类 token ([CLS]). 与 token 对应的最终隐藏状态用作分类任务的聚合序列表示(aggregate sequence representation). 句子对(Sentence Pair) 同样的, 会被整合进单独的一个序列. 为了能够将文本中的句子进行区分, BERT使用了两种方式, 首先通过一个特殊 token ([SEP]) 来进行区分, 然后加入一个经过学习的 embedding 来判断每一个 token 属于 sentence A 还是 sentence B.

已知一个token的情况下, 它对应的输入表征为 token 本身, 它对应的 segment以及 position embedding 的累加. 如下图所示.

![](https://bbs-img.huaweicloud.com/blogs/img/paperfig-2.PNG)
BERT Input Respresentation

### 预训练BERT

BERT使用了 BooksCorpus(800M 字)和英文维基百科(2,500M 字)做为预训练语料库

#### Task #1: MLM (Masked Language Model)

直观来讲,深度双向模型比从左到右模型或从左到右和+右到左模型的浅层级联更强大. 但是,标准的 conditional language models只能从左到右或从右到左进行训练, 因为双向条件会允许每个词间接"看到自己", 也就是说, 这样的条件会允许模型"作弊", 使其在一个多层语境中可以轻松预测目标词.

为了解决这样的问题, BERT随机对一部分输入的 token 采取了 `mask` 使其不可见. BERT对每一个序列采取了15%的mask, 并**只用于预测被mask掉的单词, 而不是预测整个输入**

但是这样做会导致另外一个问题: 这样的训练方式会导致在 fine tuning 过程中出现模型间的差异性, 因为在 fine tuning 过程中不会出现  [MASK] token. 为了解决这个问题, BERT在预训练过程中并不总是对15%的 token 采取 mask, 而是以80%概率进行mask , 10%会随机选取一个token代替, 最后剩下的10% 会保留原有token. 接着, $T_i$ 则会用于预测实际token (结合 corss entropy loss)

#### Task #2: NSP (Next Sentence Prediction)

传统的机器问答 (Question Answering) 和自然语言推理 (Natural Language Inference) 任务一般是基于两个句子之间的关系, 也就是说, 这样的方式无法捕获到基于 Language Model训练得到的信息. 为了训练一个理解句子关系的模型, BERT 针对Binarized next sentence prediction任务进行了预训练. 当为每个预训练示例选择句子 A 和 B 时, 50% 的时间 B 是 A 之后的实际下一个句子(标记为 IsNext), 50% 的时间它是来自语料库的随机句子(标记为作为 NotNext).

### Fine-tuning BERT

对于涉及文本对的(text pair)的应用场景,一个常见的模式是在应用双向交叉注意力 (bidirectional cross attention) 之前对文本对进行独立编码.相反,BERT 使用自注意力机制来统一这两个阶段,因为编码具有自注意力的连接文本对可以有效地包含了两个句子之间的双向交叉注意力.对于每个任务,只需将特定于任务的输入和输出插入 BERT 并微调所有参数.

在输出端,token representation 放入输出层用于标记级任务,例如序列标记或问答,而 [CLS] 表示被输入到输出层进行分类,例如蕴含或情感分析.
