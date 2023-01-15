---
title: Installing minconda on Linux machine
summary: Introductions on how to use command line to install miniconda
author: Junxiao Guo
date: 2021-01-01
tags:
  - python
  - environment
  - conda
---

```shell
wget -c https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
# For linux users

# For mac users
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh

```

```shell
chmod 777 Miniconda3-latest-Linux-x86_64.sh # grant permission
bash Miniconda3-latest-Linux-x86_64.sh # execute
Do you wish the installer to initialize Miniconda3
by running conda init? [yes|no]
```

```shell
ubuntu@ubuntu:~/miniconda3/bin$ chmod 777 activate 
ubuntu@ubuntu:~/miniconda3/bin$ source ./activate 
(base) ubuntu@ubuntu:~/miniconda3/bin$ 
```

```shell
# Configure mirror settings
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/bioconda/
```