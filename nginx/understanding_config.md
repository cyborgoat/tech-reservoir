# Understanding NGINX configurations

## Build a virtual host

`demo-site.zip` is a sample file to conf host.

`nginx.conf.bak` contains the original backup file.

## file

go to `/etc/nginx/nginx.conf`

```conf
user root;

events{}

http{
    include mime.types;

    server{
        listen 80;
        server_name 143.244.190.79;
        root /sites/demo/;
    }
}
```

```shell
$ nginx -t # Check if the modification is valid
$ systemctl reload nginx
$ curl -I http://143.244.190.79/style.css  # check css style type
```