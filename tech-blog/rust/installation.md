---
title: Rust Installation Tutorial 
summary: This article goes through how to install cargo on you local machine and tutorial on proxy settings if you are located in China.
author: Junxiao Guo
date: 2023-01-19
tags:
  - rust
  - programming-language
  - installation
---

Rust is a multi-paradigm, high-level, general-purpose programming language. Rust emphasizes performance, type safety, and concurrency. Rust enforces memory safety—that is, that all references point to valid memory—without requiring the use of a garbage collector or reference counting present in other memory-safe languages.

This article goes through how to install cargo on you local machine and tutorial on mirror settings if you are located in China.

## Installation

The official Rust document website is [https://www.rust-lang.org/](https://www.rust-lang.org/)

### Rustup: the Rust installer and version management tool

The primary way that folks install Rust is through a tool called Rustup, which is a Rust installer and version management tool.

It looks like you’re running macOS, Linux, or another Unix-like OS. To download Rustup and install Rust, run the following in your terminal, then follow the on-screen instructions. See "Other Installation Methods" if you are on Windows.

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

If you need to update your Rust, you can simply enter `rustup update`.

### Mirror settings for users in China

1. Go to folder  `/Users/{yourname}/.cargo/` (create one if you don't have it yet)
2. Create a file named `config.toml`
3. Add the following content to `config.toml`

```toml
[source.crates-io]
# Specify the mirror
replace-with = 'tuna' # You can replace tuna by ustc or sjtu or rustcc

# 中国科学技术大学
[source.ustc]
registry = "git://mirrors.ustc.edu.cn/crates.io-index"

# 上海交通大学
[source.sjtu]
registry = "https://mirrors.sjtug.sjtu.edu.cn/git/crates.io-index"

# 清华大学
[source.tuna]
registry = "https://mirrors.tuna.tsinghua.edu.cn/git/crates.io-index.git"

# rustcc社区
[source.rustcc]
registry = "https://code.aliyun.com/rustcc/crates.io-index.git"
```
