train_file='xxx/multiWOZ-new/train/seq.in'
mask_input='xxx/multiWOZ-new/mask.out'
mask_output='xxx/multiWOZ-new/mix/mask.in'
classify_output='xxx/multiWOZ-new/classify/multi_classify.out'
classify_input='xxx/multiWOZ-new/classify/multi_classify.in'
char_file='xxx/multiWOZ-new/char_aug/seq.in'

export CUDA_VISIBLE_DEVICES=0; python t5_main.py \
    --model_name_or_path t5-base \
    --model_save_path ./models/pretrain \
    --dataset multiwoz \
    --noise_path ${char_file} \
    --clean_path ${train_file} \
    --mask_output_path ${mask_output} \
    --mask_input_path ${mask_input} \
    --classify_output_path ${classify_output} \
    --classify_input_path ${classify_input} \
    --batch_size 8 \
    --early_stop \
    --force_del \
    --pretrain \
    --auxiliary_without_eval \