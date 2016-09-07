#!/bin/sh
DATA_FILES='Nottingham Piano-midi.de.pickle all'
for DATA_FILE in $DATA_FILES
  do
  python nade_like.py $DATA_FILE'.pickle' save_models/nade/'DATA_FILE'_nade_like.ckpt
done
