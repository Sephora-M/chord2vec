#!/bin/sh
python main.py --num_layers=1 --num_units=512  --max_epochs=50 --batch_size=64 --train_dir 'layer1_units512_noattention' --noattention --steps_per_checkpoint=100
python main.py --num_layers=2 --num_units=512  --max_epochs=50 --batch_size=64 --train_dir 'layer2_units512_noattention' --noattention --steps_per_checkpoint=100
python main.py --num_layers=1 --num_units=512  --max_epochs=50 --batch_size=64 --train_dir 'layer1_units512_attention' --attention --steps_per_checkpoint=100
python main.py --num_layers=2 --num_units=512  --max_epochs=50 --batch_size=64 --train_dir 'layer2_units512_attention' --attention --steps_per_checkpoint=100
python main.py --num_layers=1 --num_units=1024  --max_epochs=50 --batch_size=64 --train_dir 'layer1_units1024_noattention' --noattention --steps_per_checkpoint=100
python main.py --num_layers=2 --num_units=1024  --max_epochs=50 --batch_size=64 --train_dir 'layer2_units1024_noattention' --noattention --steps_per_checkpoint=100
python main.py --num_layers=1 --num_units=1024  --max_epochs=50 --batch_size=64 --train_dir 'layer1_units1024_attention' --attention --steps_per_checkpoint=100
python main.py --num_layers=2 --num_units=1024  --max_epochs=50 --batch_size=64 --train_dir 'layer2_units1024_attention' --attention --steps_per_checkpoint=100