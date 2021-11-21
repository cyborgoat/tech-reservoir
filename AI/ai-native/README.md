# AI-Native

## Background

### GuassDB AI-Native

作为全球首款AI-Native数据库，GaussDB有两大革命性突破：

第一，首次将人工智能技术融入分布式数据库的全生命周期，实现自运维、自管理、自调优、故障自诊断和自愈。在交易、分析和混合负载场景下，基于最优化理论，首创基于深度强化学习的自调优算法，调优性能比业界提升60%以上；

第二，通过异构计算创新框架充分发挥X86、ARM、GPU、NPU多种算力优势，在权威标准测试集TPC-DS上，性能比业界提升50%，排名第一。

此外，GaussDB支持本地部署、私有云、公有云等多种场景。在华为云上，GaussDB为金融、互联网、物流、教育、汽车等行业客户提供全功能、高性能的云上数据仓库服务。

## Design

### Input

- Raw Log Message
- Task Type
  - Status Report
  - Calculation Result
  - 
- Key Values
  - Sentence Abstraction
  - Key values

### Variables - Task Classification

- Predictions
- Scoring
- Abnormal Detection

### Abilities

- System general health check
  - Locate the unhealthy part by log information
  - Able to classify the problem type
  - Recommend fix suggestions.
- Abnormal Log Detections
  - Abnormal traffic
  - Abnormal log information
    - Return possible reasons. (Large text conversation model & expert experience)
  - Abnormal log sequence
    - Offer expected log sequences
  - 
- Traffic prediction
- 