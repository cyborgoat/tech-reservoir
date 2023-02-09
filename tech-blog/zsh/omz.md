---
title: Install oh-my-zsh in Ubuntu 20.04 
summary: oh-my-zsh is an open-source framework for managing ZSH configuration and is community-driven.
author: Junxiao Guo
date: 2023-02-01
tags:
  - linux
  - zsh
---

## Prerequisites

- Zsh should be installed (v4.3.9 or more recent would do but we prefer 5.0.8 and newer).
- Curl or Wget should be installed.
- Git should be installed (v2.4.11 or higher recommended).

## Installing oh-my-zsh in Ubuntu Linux

Installation of Oh My Zsh can be done by using “Curl” or “Wget” commands in your terminal. Make sure either of one utility is installed in the OS, if not install them along with git by running the following `apt` command.

```bash
$ sudo apt install curl wget git
```

Then we install `zsh` by,

```bash
$ sudo apt install zsh
```

Make sure `zsh` is correctly installed,

```bash
$ zsh --version
```

Installing ZSH will not modify and set it as the default shell. We have to modify the settings to make ZSH our default shell. Use the “chsh” command with '-s' flag to switch the default shell for the user.

```bash
$ echo $SHELL
$ chsh -s $(which zsh) 
or 
$ chsh -s /usr/bin/zsh
```

Next, install **Oh My Zsh** via the command-line with either curl or wget as shown.

```bash
$ sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"
```

or,

```bash
$ sh -c "$(wget https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh -O -)"
```

Congratulations! You have successfully installed Oh My Zsh on your ubuntu machine!
