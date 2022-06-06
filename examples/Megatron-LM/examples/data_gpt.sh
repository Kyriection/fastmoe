python tools/preprocess_data.py \
    --input bookcorpus.json \
    --output-prefix my-gpt2 \
    --vocab gpt2-vocab.json \
    --dataset-impl mmap \
    --tokenizer-type GPT2BPETokenizer \
    --merge-file gpt2-merges.txt \
    --append-eod \
    --workers 8 \
    --chunk-size 8