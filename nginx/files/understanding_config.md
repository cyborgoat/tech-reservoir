# Understanding NGINX configurations

## Build a virtual host

`demo-site.zip` is a sample file to conf host.

`nginx.conf.bak` contains the original backup file.

## file

go to `/etc/nginx/nginx.conf`

```shell
$ nginx -t # Check if the modification is valid
$ systemctl reload nginx
$ curl -I http://143.244.190.79/style.css  # check css style type
```

## Basic commands

```shell
$ ps aux | grep nginx # Check process
$ nproc # Check number of cpus
$ lscpu # Cehck cpu info
$ ulimit # maximum connections can handle

```