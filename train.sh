#!/bin/bash
while true;do
    python 16/train.py
    sleep 10s
    rsync -a --del 16/data 16/data_bak
done
