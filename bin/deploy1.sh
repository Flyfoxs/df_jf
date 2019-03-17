#!/usr/bin/env bash
cd "$(dirname "$0")"

cd ..

rsync -av   $(pwd)/imp/* hdpsbp@sbpmed-prd0"$1":/appdata/felix/df_jf/imp/


rsync -av   $(pwd)/cache/* hdpsbp@sbpmed-prd0"$1":/appdata/felix/df_jf/cache/


rsync -av --exclude-from './bin/exclude.txt' $(pwd) hdpsbp@sbpmed-prd0"$1":/appdata/felix/


date

