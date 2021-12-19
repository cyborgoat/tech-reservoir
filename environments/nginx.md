# NGINX

## Install

### Second Method (Preferred Way)

- go to https://nginx.org/en/download.html
- copy link of mainline version

```shell
$ apt-get update
$ wget https://nginx.org/download/nginx-1.21.4.tar.gz # Download the file
$ tar -zxvf nginx-1.21.4.tar.gz # Unzip the file
$ apt-get install build-essential # Download the essentials for configuration build
$ ./configure # Check for configurations
$ apt-get install libpcre3 libpcre3-dev zlib1g zlib1g-dev libssl-dev # Install necessary files to check configurations files

```

### First Method (Not recommended)

```shell
$ apt-get update
$ apt-get install nginx
$ ipconfig
$ ls -l /etc/nginx # check for installation
$ ps aux | grep nginx # check for processes
$ service nginx start # start nginx
```

## Building nginx from source

nginx.org/en/docs/configure.html

```shell
$./configure --sbin-path=/usr/bin/nginx --conf-path=/etc/nginx/nginx.conf --error-log-path=/var/log/nginx/error.log --http-log-path=/var/log/nginx/access.log --with-pcre --pid-path=/var/run/nginx.pid --with-http_ssl_module
$ make
$ make install
$ ls -l /etc/nginx # Check if the executable are installed successfully
$ nginx -V # Check version & arguments
$ nginx -t # Check if configuration is valid
$ nginx # start nginx
```

## Adding an NGINX  service

### Start & Stop

```shell
$ ps aux | grep nginx # check for process
$ nginx -h # for help
$ nginx -s stop # Stop the nginx
$ nginx 
```

### systemD service

**Enables much much mor commands for nginx**

Check [nginx init script resource page](https://www.nginx.com/resources/wiki/start/topics/examples/initscripts/)

got to the Systemd and navigate to the [system service file](https://www.nginx.com/resources/wiki/start/topics/examples/systemd/)

The instruction tells you the location that existing for the service config

```shell
$ touch /lib/systemd/system/nginx.service
$ vim /lib/systemd/system/nginx.service
```

Add the following context to configure the nginx service

```txt
[Unit]
Description=The NGINX HTTP and reverse proxy server
After=syslog.target network-online.target remote-fs.target nss-lookup.target
Wants=network-online.target

[Service]
Type=forking
PIDFile=/var/run/nginx.pid
ExecStartPre=/usr/bin/nginx -t
ExecStart=/usr/bin/nginx
ExecReload=/bin/kill -s HUP $MAINPID
ExecStop=/bin/kill -s QUIT $MAINPID
PrivateTmp=true

[Install]
WantedBy=multi-user.target
```

```shell
$ systemctl start nginx
$ ps aux | grep nginx
$ systemctl status nginx # systemD manager to check the formal status
$ systemctl stop nginx # Stop the nginx
$ systemctl enable nginx # Allows nginx start as the system reboot
$ reboot # reboot the linux machine
$ systemctl status nginx
```
