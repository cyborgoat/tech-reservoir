# NGINX

## Install

```shell
apt-get update
apt-get install nginx
```

- go to https://nginx.org/en/download.html
- copy link of mainline version

```shell
wget https://nginx.org/download/nginx-1.21.4.tar.gz # Download the file
tar -zxvf nginx-1.21.4.tar.gz # Unzip the file
apt-get install build-essential # Download the essentials for configuration build
./configure # Check for configurations
apt-get install libpcre3 libpcre3-dev zlib1g zlib1g-dev libssl-dev # Install necessary files to check configurations files

```
