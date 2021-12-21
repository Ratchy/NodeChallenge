#!/usr/bin/env bash
SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"
#SCRIPTPATH="."
echo "Path: $SCRIPTPATH"
docker build -t nodulegenerator "$SCRIPTPATH"
