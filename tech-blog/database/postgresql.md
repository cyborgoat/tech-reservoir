---
title: Postgres Database Installation
summary: A brief summary of commands to install postgres database
author: Junxiao Guo
date: 2021-01-01
tags:
  - database
  - postgres
---

## Installation

For ubuntu 20.04

```shell
sudo apt update
sudo apt install postgresql postgresql-contrib
sudo systemctl start postgresql.service

```

For Mac Users

```shell
brew install postgresql
```

## Step 2 — Using PostgreSQL Roles and Databases

```shell
sudo -i -u postgres
psql
createuser --interactive
createdb sammys # Create a database
sudo adduser sammy # Create a user
psql -d postgres
\conninfo
```
