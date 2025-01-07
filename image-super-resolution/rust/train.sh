#!/bin/bash
DATA="/blok/@data"
# DATA="$HOME/Data"
WCM="$DATA/org/wikimedia/wikimedia-commons-merged4"
# RELEASE="--release"
RELEASE=""
RUST_BACKTRACE=full exec cargo run $RELEASE -- train $WCM/s50/train/small/ $WCM/s50/train/large $WCM/s50/valid/small/ $WCM/s50/valid/large/ ./runs/3-hugetrain-resnet 2
