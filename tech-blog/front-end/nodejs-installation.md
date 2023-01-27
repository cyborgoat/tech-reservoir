---
title: Node.js Installation For Ubuntu 20.04
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

## Installing Node.js with Apt from the Default Repositories

Ubuntu 20.04 contains a version of Node.js in its default repositories that can be used to provide a consistent experience across multiple systems. At the time of writing, the version in the repositories is 10.19. This will not be the latest version, but it should be stable and sufficient for quick experimentation with the language.

To get this version, you can use the apt package manager. Refresh your local package index first:

```bash
sudo apt update
```

```bash
sudo apt install nodejs
```

Check that the install was successful by querying node for its version number:

```bash
$ node -v
> v10.19.0
```

If the package in the repositories suits your needs, this is all you need to do to get set up with Node.js. In most cases, you’ll also want to also install npm, the Node.js package manager. You can do this by installing the npm package with apt:

```bash
$ sudo apt install npm
```

This allows you to install modules and packages to use with Node.js.

At this point, you have successfully installed Node.js and npm using apt and the default Ubuntu software repositories. The next section will show how to use an alternate repository to install different versions of Node.js.

## Mirror Setting

For users in China, please add the following mirror settings to make sure your `npm` running as expected

```bash
npm config set registry https://registry.npmirror.com
```
