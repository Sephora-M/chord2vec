#!/bin/sh
BATCH_SIZE=128
STEPS_CKPT=100
MAX_EPOCHS=20
#DATA_FILES='MuseData Nottingham'
DATA_FILES='JSB_Chorales'
for NUM_UNITS in 512 1024
   do
   for NUM_LAYER in 3 2 1
   	do  
   	python main.py --num_layers=$NUM_LAYER --num_units=1024  --max_epochs=$MAX_EPOCHS --batch_size=$BATCH_SIZE --train_dir $DATA_FILE'_'$NUM_LAYER'_'$NUM_UNITS'layers_attention' --attention --steps_per_checkpoint=$STEPS_CKPT --data_file $DATA_FILE'.pickle'
   	python main.py --num_layers=$NUM_LAYER --num_units=1024  --max_epochs=$MAX_EPOCHS --batch_size=$BATCH_SIZE --train_dir $DATA_FILE'_'$NUM_LAYER'_'$NUM_UNITS'layers' --noattention --steps_per_checkpoint=$STEPS_CKPT --data_file $DATA_FILE'.pickle'
   done
done