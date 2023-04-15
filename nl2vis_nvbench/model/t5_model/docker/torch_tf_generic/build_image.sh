#!/bin/bash


# Variables
DIVIDER_LINE="\n*******************************************************************************\n"
CURRENT_DIR=$PWD
IMAGE_NAME="w266-torch-tf-generic:latest"

#--------------------------------------------------------------------------------------------------
printf $DIVIDER_LINE
printf "::BUILD IMAGE::\n\n"

docker build -t $IMAGE_NAME .