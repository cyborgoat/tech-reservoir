# Zsh & oh my zsh

```shell
$ sudo apt-get install zsh
$ zsh --version
$ chsh -s /bin/zsh
$ sudo apt-get install git
$ sudo apt-get install wget
$ sh -c "$(wget https://raw.github.com/ohmyzsh/ohmyzsh/master/tools/install.sh -O -)"

```

## setup missing fonts(powerline)
```shell
cd
git clone https://github.com/powerline/fonts.git
cd fonts
./install.sh
cd ..
rm -rf fonts
```

```shell
wget https://raw.githubusercontent.com/powerline/powerline/develop/font/10-powerline-symbols.conf
 
wget https://raw.githubusercontent.com/powerline/powerline/develop/font/PowerlineSymbols.otf
 
sudo mkdir /usr/share/fonts/OTF
 
sudo cp 10-powerline-symbols.conf /usr/share/fonts/OTF/
 
sudo mv 10-powerline-symbols.conf /etc/fonts/conf.d/
 
sudo mv PowerlineSymbols.otf /usr/share/fonts/OTF/
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