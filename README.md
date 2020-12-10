
<br />
<p align="center">
  <a href="https://github.com/heraclex12/VLSP2020-Fake-News-Detection">
  </a>

  <h3 align="center">VNPunc: Vietnamese Punctuation Prediction using Pretrained Language Models</h3>

  <p align="center">
    Fine-tune a variety of pre-trained Transformer-based models to solve Vietnamese Punctuation Prediction task.
    <br />
  </p>
</p>



<!-- ABOUT THE PROJECT -->
## About The Project
In this project, we utilize the effectiveness of the different pre-trained language models such as vELECTRA, vBERT, XLM-RoBERTa to restore seven punctuation marks in Vietnamese datasets.

We also stack a LSTM layer and CRF layer on the top of output representations. This contributions achieve a significant improvement over the previous models.

### Prerequisites

To reproduce the experiment of our model, please install the requirements.txt according to the following instructions:
* huggingface transformer
* pytorch
* python3
```sh
pip install -r requirements.txt
```

### Data

We also include Vietnamese novel and news dataset in this project. Thanks to [this work](https://github.com/BinhMisfit/vietnamese-punctuation-prediction) for providing this datasets.

## Instructions

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

<!-- CONTACT -->
## Contact

Hieu Tran - heraclex12@gmail.com

Project Link: [https://github.com/heraclex12/VN-Punc-Pretrained-LMs](https://github.com/heraclex12/VN-Punc-Pretrained-LMs)



<!-- ACKNOWLEDGEMENTS -->
## Acknowledgements
* [Improving Sequence Tagging for Vietnamese Text Using Transformer-based Neural Models](https://arxiv.org/abs/2006.15994)
* [PhoBERT: Pretrained language model for Vietnamese](https://github.com/VinAIResearch/PhoBERT)
* [Unsupervised Cross-lingual Representation Learning at Scale](https://arxiv.org/pdf/1911.02116.pdf)
