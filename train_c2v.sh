#!/bin/sh
BATCH_SIZE=128
STEPS_CKPT=100
MAX_EPOCHS=20
DATA_FILE='MuseData'
#DATA_FILE='Nottingham'
#DATA_FILE='Piano-midi.de'
#DATA_FILE='JSB_Chorales'
python main.py --num_layers=3 --num_units=1024  --max_epochs=$MAX_EPOCHS --batch_size=$BATCH_SIZE --train_dir $DATA_FILE'_3layersattention' --attention --steps_per_checkpoint=$STEPS_CKPT ----data_file $DATA_FILE'.pickle'
python main.py --num_layers=2 --num_units=1024  --max_epochs=$MAX_EPOCHS --batch_size=$BATCH_SIZE --train_dir $DATA_FILE'_2layers' --noattention --steps_per_checkpoint=$STEPS_CKPT ----data_file $DATA_FILE'.pickle'
python main.py --num_layers=3 --num_units=1024  --max_epochs=$MAX_EPOCHS --batch_size=$BATCH_SIZE --train_dir $DATA_FILE'_3layers' --noattention --steps_per_checkpoint=$STEPS_CKPT ----data_file $DATA_FILE'.pickle'
python main.py --num_layers=2 --num_units=1024  --max_epochs=$MAX_EPOCHS --batch_size=$BATCH_SIZE --train_dir $DATA_FILE'_2layersattention' --attention --steps_per_checkpoint=$STEPS_CKPT ----data_file $DATA_FILE'.pickle'
#python main.py --num_layers=3 --num_units=512  --max_epochs=$MAX_EPOCHS --batch_size=$BATCH_SIZE --train_dir $DATA_FILE --noattention --steps_per_checkpoint=$STEPS_CKPT ----data_file $DATA_FILE'.pickle'
#python main.py --num_layers=3 --num_units=1024  --max_epochs=$MAX_EPOCHS --batch_size=$BATCH_SIZE --train_dir $DATA_FILE --noattention --steps_per_checkpoint=$STEPS_CKPT ----data_file $DATA_FILE'.pickle'
#python main.py --num_layers=2 --num_units=512  --max_epochs=$MAX_EPOCHS --batch_size=$BATCH_SIZE --train_dir 'all_data' --noattention --steps_per_checkpoint=$STEPS_CKPT --all_data_sets
#python main.py --num_layers=1 --num_units=512  --max_epochs=$MAX_EPOCHS --batch_size=$BATCH_SIZE --train_dir 'layer1_units512_attention' --attention --steps_per_checkpoint=$STEPS_CKPT
#python main.py --num_layers=2 --num_units=512  --max_epochs=$MAX_EPOCHS --batch_size=$BATCH_SIZE --train_dir 'layer2_units512_attention' --attention --steps_per_checkpoint=$STEPS_CKPT
#python main.py --num_layers=1 --num_units=1024  --max_epochs=$MAX_EPOCHS --batch_size=$BATCH_SIZE --train_dir 'layer1_units1024_noattention' --noattention --steps_per_checkpoint=$STEPS_CKPT
#python main.py --num_layers=2 --num_units=1024  --max_epochs=$MAX_EPOCHS --batch_size=$BATCH_SIZE --train_dir 'layer2_units1024_noattention' --noattention --steps_per_checkpoint=$STEPS_CKPT
#python main.py --num_layers=1 --num_units=1024  --max_epochs=$MAX_EPOCHS --batch_size=$BATCH_SIZE --train_dir 'layer1_units1024_attention' --attention --steps_per_checkpoint=$STEPS_CKPT
#python main.py --num_layers=2 --num_units=1024  --max_epochs=$MAX_EPOCHS --batch_size=$BATCH_SIZE --train_dir 'layer2_units1024_attention' --attention --steps_per_checkpoint=$STEPS_CKPT