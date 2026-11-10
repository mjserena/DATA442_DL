#!/bin/bash

scp \
    'bora:/sciclone/home/mjserena/data442_DL/logs/*.out' \
    'bora:/sciclone/home/mjserena/data442_DL/*.csv' \
    'bora:/sciclone/home/mjserena/data442_DL/*.pt' \
    'bora:/sciclone/home/mjserena/data442_DL/*.txt' \
    /Users/mjserena/Downloads/

