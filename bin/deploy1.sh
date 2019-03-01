#!/usr/bin/env bash
cd "$(dirname "$0")"

cd ..

rsync -av --exclude-from './bin/exclude.txt' $(pwd) hdpsbp@sbpmed-prd0"$1":/appdata/felix/


date

