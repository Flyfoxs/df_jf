#!/usr/bin/env bash
cd "$(dirname "$0")"

cd ..

if [[ -z "$1" ]]; then
    rsync -av --exclude-from './bin/exclude.txt' $(pwd) hdpsbp@ai-prd-07:/users/hdpsbp/felix/
else
    rsync -av $(pwd) hdpsbp@ai-prd-05:/users/hdpsbp/felix/
fi

date

#rsync -av  ./output/0.70180553000.csv hdpsbp@ai-prd-07:/users/hdpsbp/felix/output