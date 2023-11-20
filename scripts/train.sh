train_file='xxx/multiWOZ-new/train/train.txt'
data_path='xxx/multiWOZ-new'
test_path='xxx/multiWOZ-new/test'
valid_file='xxx/multiWOZ-new/valid/valid.txt'
valid_demons_input="xxx/multiWOZ-new/valid/valid_demons_2.in"
valid_demons_output="xxx/multiWOZ-new/valid/valid_demons_2.out"
logging_path='/your_logging_path'
mix_input='xxx/multiWOZ-new/mix/mix_all_demons_num_2_noise.in'
mix_output='xxx/multiWOZ-new/mix/mix_all_demons_num_2_noise.out'
classify_path='xxx/multiWOZ-new/classify'
char_path='xxx/multiWOZ-new/char_aug'

export CUDA_VISIBLE_DEVICES=0; python t5_main.py \
    --seed 2023 \
    --model_name_or_path t5-base \
    --model_save_path ./models/noise_demos \
    --dataset multiwoz \
    --training \
    --metric f1 \
    --train_file_path ${train_file} \
    --dev_file_path ${valid_file} \
    --add_demonstration \
    --demons_train_path ${mix_input} \
    --demons_out_path ${mix_output} \
    --demons_valid_path ${valid_demons_input} \
    --demons_val_out_path ${valid_demons_output} \
    --max_seq_length 256 \
    --decode_max_seq_length 256 \
    --eval_batch_size 16 \
    --test_file_path ${test_path} \
    --logging_file_path ${logging_path} \
    --num 2 \
    --batch_size 8 \
    --num_epochs 1 \
    --learning_rate 5e-5 \
    --evaluation_steps 500 \
    --early_stop \
    --force_del \
    --add_demonstration \
    --continue_training_path ./models/pretrain \