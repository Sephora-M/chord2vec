#!/bin/sh
BATCH_SIZE=128
STEPS_CKPT=100
MAX_EPOCHS=20
#DATA_FILES='MuseData Nottingham'
#DATA_FILES='Piano-midi.de JSB_Chorales'
for DATA_FILE in $DATA_FILES
   do
   for NUM_LAYER in 3 2
   	do  
   	python main.py --num_layers=$NUM_LAYER --num_units=1024  --max_epochs=$MAX_EPOCHS --batch_size=$BATCH_SIZE --train_dir $DATA_FILE'_'$NUM_LAYER'layers_attention' --attention --steps_per_checkpoint=$STEPS_CKPT --data_file $DATA_FILE'.pickle'
   	python main.py --num_layers=$NUM_LAYER --num_units=1024  --max_epochs=$MAX_EPOCHS --batch_size=$BATCH_SIZE --train_dir $DATA_FILE'_'$NUM_LAYER'layers' --noattention --steps_per_checkpoint=$STEPS_CKPT --data_file $DATA_FILE'.pickle'
   done
done
