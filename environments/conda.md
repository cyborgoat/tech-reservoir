```shell
wget -c https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
# 这个版本是适合于linux的，要看清楚噢。

# mac用户请用：
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh

# mac用户选择图形化界面的anaconda版本也可。
# 传送门：https://www.anaconda.com/products/individual 
# 往下翻，选择64-Bit Graphical Installer
```

```shell
chmod 777 Miniconda3-latest-Linux-x86_64.sh #给执行权限
bash Miniconda3-latest-Linux-x86_64.sh #运行
Do you wish the installer to initialize Miniconda3
by running conda init? [yes|no]
```

```shell
ubuntu@ubuntu:~/miniconda3/bin$ chmod 777 activate 
ubuntu@ubuntu:~/miniconda3/bin$ source ./activate 
(base) ubuntu@ubuntu:~/miniconda3/bin$ 
```

```shell
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/bioconda/
```