#!/bin/sh
DATA_FILES='MuseData Nottingham Piano-midi.de.pickle all'
for DATA_FILE in $DATA_FILES
  do
  python nade_like.py $DATA_FILE save_models/'DATA_FILE'_nade_like.ckpt
done