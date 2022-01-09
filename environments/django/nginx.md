# How to use gunicorn & nginx

```shell
$ touch /etc/nginx/sites-available/project_name
$ cat project_name.conf > /etc/nginx/sites-available/project_name
$ cd /etc/nginx/sites-enabled/
$ sudo ln -s /etc/nginx/sites-available/project_name
```