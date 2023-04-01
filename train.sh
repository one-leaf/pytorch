#!/bin/bash
while true;do
    python 16/train.py
    sleep 10s
    rsync -avP --del 16/data 16/data_bak
done
