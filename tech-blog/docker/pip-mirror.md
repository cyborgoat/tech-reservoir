---
title: Add pip mirror settings for dockerfile
summary: This article teaches how to add pip mirrors inside docker file
author: Junxiao Guo
date: 2023-01-30
tags:
  - docker
---

## Install python and pip

```dockerfile
RUN apt-get update
RUN apt-get install -y python3.5  
RUN apt-get install -y python3-pip 
```

## Update pip to latest

```dockerfile
RUN pip3 install pip -U
```

## Config mirror settings

```dockerfile
RUN pip3 config set global.index-url http://mirrors.aliyun.com/pypi/simple
RUN pip3 config set install.trusted-host mirrors.aliyun.com
```

> If you want to use other mirror resources, simply replace the url after `index-url`