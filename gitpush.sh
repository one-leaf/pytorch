#!/bin/sh

# 进行git的函数，处理从add到push
function gitpush()
{
    error_str1="fatal"
    error_str2="error"

    # push
    while true
    do
        echo "***开始push本地仓库***"
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
    done
}

gitpush
