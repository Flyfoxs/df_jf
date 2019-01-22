#!/usr/bin/env bash
cd "$(dirname "$0")"

cd ..

if [[ -z "$1" ]]; then
    rsync -av --exclude-from './bin/exclude.txt' $(pwd) hdpsbp@ai-prd-07:/users/hdpsbp/bk/
else
    rsync -av $(pwd) hdpsbp@ai-prd-07:/users/hdpsbp/bk/
fi



