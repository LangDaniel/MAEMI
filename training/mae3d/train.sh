#!/bin/bash

POSITIONAL_ARGS=()
FOREGROUND=0
CPU="0-19"

while [[ $# -gt 0 ]]; do
  case $1 in
    -c|--cpu-list)
      CPU="$2"
      shift # past argument
      shift # past value
      ;;
    -g|--gpu)
      SEARCHPATH="$2"
      shift # past argument
      shift # past value
      ;;
    -fg)
      FOREGROUND=1
      shift # past argument
      ;;
    -*|--*)
      echo "Unknown option $1"
      exit 1
      ;;
    *)
      POSITIONAL_ARGS+=("$1") # save positional arg
      shift # past argument
      ;;
  esac
done

set -- "${POSITIONAL_ARGS[@]}" # restore positional parameters

source activate mae 
if [ ${FOREGROUND} -eq 1 ]; then
        taskset -c ${CPU} python $1 $2
else
        nohup taskset -c ${CPU} python $1 $2 &
fi
printf "$1: $!\n" >> log 
conda deactivate

