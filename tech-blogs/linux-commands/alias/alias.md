# Alias

1. create ~/.bash_aliases

```shell
alias getdates='data | tee /home/cyborgoat/fulldate.txt | cut --delimiter=" " --fields=1 | tee /home/cyborgoat/shortdate.txt | xarg echo hello'
```

2. restart terminal will make the alias work.