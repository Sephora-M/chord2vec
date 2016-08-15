#!/bin/sh
STEPS_CKPT=200
MAX_EPOCHS=20
#DATA_FILES='MuseData Nottingham'
DATA_FILE='JSB_Chorales'
for NUM_UNITS in 512 1024
do
for NUM_LAYER in 2 1
 do
  for BATCH_SIZE in 64 128 
   do  
    python main.py --num_layers=$NUM_LAYER --num_units=$NUM_UNITS  --max_epochs=$MAX_EPOCHS --batch_size=$BATCH_SIZE --train_dir 'save_models/'$NUM_LAYER'layers_models/GD/'$DATA_FILE'_'$BATCH_SIZE'batch_'$NUM_LAYER'layers_'$NUM_UNITS'units_attention' --attention --steps_per_checkpoint=$STEPS_CKPT --data_file $DATA_FILE'.pickle' --GD --learning_rate=0.5
    python main.py --num_layers=$NUM_LAYER --num_units=$NUM_UNITS  --max_epochs=$MAX_EPOCHS --batch_size=$BATCH_SIZE --train_dir 'save_models/'$NUM_LAYER'layers_models/GD/'$DATA_FILE'_'$BATCH_SIZE'batch_'$NUM_LAYER'layers_'$NUM_UNIT'units' --noattention --steps_per_checkpoint=$STEPS_CKPT --data_file $DATA_FILE'.pickle' --GD --learning_rate=0.5
done
done 
done
