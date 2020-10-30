# VNPunc
Vietnamese Punctuation Prediction using Pretrained Language Models


## Intructions

Must install required package:
> pip3 install requirements.txt

### Training
```
python3 run_train_punc.py --bert_model=bert-base-multilingual-cased \
                            --data_dir=data/News \ 
                            --output_dir=outputs \ 
                            --task_name=punctuation_prediction \
                            --max_seq_length=190 \
                            --do_train \
                            --do_eval  \ 
                            --eval_on=test \
                            --train_batch_size=8 \ 
                            --model_type=punc_bert
```