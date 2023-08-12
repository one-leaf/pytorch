#!/bin/bash

# 进行git的函数，处理从add到push
function gitpush()
{
    error_str1="fatal"
    error_str2="error"

    # pull
    while true
    do
        echo "***开始pull本地仓库***"
        git config --global http.https://github.com.proxy socks5://127.0.0.1:10086

        var=$(git pull 2>&1)
        if [[ $var =~ $error_str1 || $var =~ $error_str2 ]]; then 
            echo "***pull远程仓库出现错误***"
            echo $var
        else
            echo $var
            echo "***pull完成***"
            break
        fi

        git config --global --unset http.https://github.com.proxy

    done
    echo `date -R`
}

gitpush
