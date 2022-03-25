# How to use gunicorn & nginx

## Installations

```shell
sudo apt-get update
sudo apt-get install python3-pip python3-dev libpq-dev postgresql postgresql-contrib nginx
sudo -u postgres psql

```

```shell
$ touch /etc/nginx/sites-available/project_name
$ cat project_name.conf > /etc/nginx/sites-available/project_name
$ cd /etc/nginx/sites-enabled/
$ sudo ln -s /etc/nginx/sites-available/project_name
```