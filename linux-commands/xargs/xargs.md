# xargs command

```shell
$ date | echo # Won't return anything
$ date | xargs echo
Sun Nov 21 22:19:45 CST 2021
$ date | xargs echo "hello"
hello Sun Nov 21 22:20:03 CST 2021
$ date | cut --delimiter=" " --fields=1 | xargs echo
Sun
$ cat filestodelte.txt | xargs rm # This will remove all the files listed in filestodelte.txt 
```