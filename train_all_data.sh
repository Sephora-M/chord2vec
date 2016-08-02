#!/bin/sh
BATCH_SIZE=128
STEPS_CKPT=100
MAX_EPOCHS=20
for NUM_LAYER in 3 2
   	do 
	python main.py --num_layers=$NUM_LAYER --num_units=1024  --max_epochs=$MAX_EPOCHS --batch_size=$BATCH_SIZE --train_dir 'all_data_'$NUM_LAYER'layers_attention' --attention --all_data_sets
	python main.py --num_layers=$NUM_LAYER --num_units=1024  --max_epochs=$MAX_EPOCHS --batch_size=$BATCH_SIZE --train_dir 'all_data_'$NUM_LAYER'layers' --noattention --all_data_sets
done