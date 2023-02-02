#!/bin/bash
set -e

if [ $# -eq 0 ] 
    then 
      echo no download requested
    else
        cd /opt/physionet
        wget -A .txt -r -nc -c -np --user $PHYSIONET_USER --password $PHYSIONET_PASSWORD https://physionet.org/files/mimic-cxr/2.0.0/files/
        seq 10 19 | parallel -j4 wget -A .jpg -r -nc -c -np --user $PHYSIONET_USER --password $PHYSIONET_PASSWORD https://physionet.org/files/mimic-cxr-jpg/2.0.0/files/p{}/
fi
