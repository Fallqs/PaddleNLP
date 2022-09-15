dir="finetune10"
src="./data_tools"

python -u data_tools/tio_gen2.py \
    --data_format JSON \
    --seq_len 1024 \
    --input_path $src/sample.jsonl \
    --output_prefix $dir/code_python  \
    --workers 113 \
    --log_interval 10000
