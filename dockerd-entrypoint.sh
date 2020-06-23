#!/bin/bash
set -e

if [[ "$1" = "serve" ]]; then
    shift 1
    torchserve --start --ts-config ./deploy/config.properties --model-store ./deploy/model_store --models medium_consonant=medium_consonant.mar
else
    eval "$@"
fi

# prevent docker exit
tail -f /dev/null