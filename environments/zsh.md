# Zsh & oh my zsh

```shell
$ sudo apt-get install zsh
$ zsh --version
$ chsh -s /bin/zsh
$ sudo apt-get install git
$ sh -c "$(curl -fsSL https://raw.github.com/robbyrussell/oh-my-zsh/master/tools/install.sh)"

```

## setup missing fonts(powerline)
```shell
cd
wget https://github.com/powerline/powerline/raw/develop/font/PowerlineSymbols.otf
wget https://github.com/powerline/powerline/raw/develop/font/10-powerline-symbols.conf
mv PowerlineSymbols.otf ~/.fonts/
mkdir -p .config/fontconfig/conf.d #if directory doesn't exists
fc-cache -vf ~/.fonts/ # Clean font cache
mv 10-powerline-symbols.conf ~/.config/fontconfig/conf.d/ # move config file

```


edit ~/.zshrc  thememe to agnoster

```txt
ZSH_THEME="agnoster"
```


## optional

```shell
$ sudo apt-get install autojump
```

```txt
vim .zshrc
#在最后一行加入，注意点后面是一个空格
. /usr/share/autojump/autojump.sh
```

```shell
$ source ~/.zshrc
```