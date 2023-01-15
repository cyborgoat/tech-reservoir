# tee command

```shell
$ date | tee fulldate.txt | cut --dilimiter=" " --field=1 | tee todaysdate.txt
```