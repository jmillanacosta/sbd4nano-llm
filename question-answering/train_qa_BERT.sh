#!/bin/bash
#rm *py
#rm *txt
modules="requirements.txt run_qa.py run_qa_beam_search.py run_qa_beam_search_no_trainer.py run_qa_no_trainer.py run_seq2seq_qa.py trainer_qa.py utils_qa.py"

#for module in $modules; do
  #wget "https://raw.githubusercontent.com/huggingface/transformers/main/examples/pytorch/question-answering/$module"
#done

export SQUAD='squad/'
python -m venv .env
source .env/bin/activate
pip install git+https://github.com/huggingface/transformers
pip install -r requirements.txt
pip install --upgrade google-api-python-client
python run_qa.py \
  --model_name_or_path bert-base-uncased \
  --dataset_name $SQUAD \
  --do_train \
  --do_eval \
  --per_device_train_batch_size 12 \
  --learning_rate 3e-5 \
  --num_train_epochs 2 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --output_dir /tmp/debug_squad/