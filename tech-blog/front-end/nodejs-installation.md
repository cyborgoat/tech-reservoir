---
title: Node.js Installation Tutorial 
summary: This article goes through how to install Node.js on you local machine and tutorial on mirror settings if you are located in China.
author: Junxiao Guo
date: 2023-01-19
tags:
  - node.js
  - frontend
---

## Installation

The official website for Node.js is [https://nodejs.org/en/](https://nodejs.org/en/)

Go to the page and download the package

## Mirror Setting

For users in China, please add the following mirror settings to make sure your `npm` running as expected

```bash
npm config set registry https://registry.npmirror.com
```