#!/bin/bash
set -e

scp -J mjserena@bastion.wm.edu \
    /Users/mjserena/Documents/GitHub/DATA442_DL/MsPacman_DL.py \
    mjserena@bora.sciclone.wm.edu:/sciclone/home/mjserena/data442_DL/

scp -J mjserena@bastion.wm.edu \
    /Users/mjserena/Documents/GitHub/DATA442_DL/run_pacman.sbatch \
    mjserena@bora.sciclone.wm.edu:/sciclone/home/mjserena/data442_DL/