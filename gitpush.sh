#!/bin/bash

# 进行git的函数，处理从add到push
function gitpush()
{
    error_str1="fatal"
    error_str2="error"

    # push
    while true
    do
        echo "***开始push本地仓库***"
        git config --global http.https://github.com.proxy socks5://127.0.0.1:10086

        var=$(git push origin master:master 2>&1)
        if [[ $var =~ $error_str1 || $var =~ $error_str2 ]]; then 
            echo "***push远程仓库出现错误***"
            echo $var
        elif [[ $var =~ "git pull" ]]; then 
            echo "***pull远程仓库***"
            var=$(git pull 2>&1)
            echo $var
        else
            echo $var
            echo "***push完成***"
            break
        fi
        git config --global --unset http.https://github.com.proxy

    done
    echo `date -R`
}

gitpush
