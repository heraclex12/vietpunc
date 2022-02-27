
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
In this project, we utilize the effectiveness of the different pre-trained language models such as viELECTRA, viBERT, XLM-RoBERTa to restore seven common punctuation marks in Vietnamese.

We also stack a LSTM layer and CRF layer on the top of output representations. This contributions achieve a significant improvement over the previous models.

### Prerequisites

To reproduce the experiments of our model, please install the `requirements.txt` according to the following instructions:
* transformers==4.16.2
* pytorch==1.10.0
* python==3.7
```sh
pip install -r requirements.txt
```

### Data

We also include Vietnamese novel and news dataset in this project. Thanks to [this work](https://github.com/BinhMisfit/vietnamese-punctuation-prediction) for providing these datasets.

## Instructions

### Training
```
python3 run_train_punc.py --model_name_or_path=bert-base-multilingual-cased \
                            --model_arch lstm_crf \
                            --model_type bert \
                            --data_dir=data/News \ 
                            --output_dir=outputs \ 
                            --task_name=punctuation_prediction \
                            --max_seq_length=190 \
                            --do_train \
                            --do_eval  \ 
                            --eval_on=test \
                            --train_batch_size=32
```

<!-- CONTACT -->
## Contact

Hieu Tran - heraclex12@gmail.com

Code for paper [An Efficient Transformer-Based Model for Vietnamese Punctuation Prediction](https://link.springer.com/chapter/10.1007/978-3-030-79463-7_5)


## Citation
```
@InProceedings{10.1007/978-3-030-79463-7_5,
      author="Tran, Hieu
      and Dinh, Cuong V.
      and Pham, Quang
      and Nguyen, Binh T.",
      editor="Fujita, Hamido
      and Selamat, Ali
      and Lin, Jerry Chun-Wei
      and Ali, Moonis",
      title="An Efficient Transformer-Based Model for Vietnamese Punctuation Prediction",
      booktitle="Advances and Trends in Artificial Intelligence. From Theory to Practice",
      year="2021",
      publisher="Springer International Publishing",
      address="Cham",
      pages="47--58",
      isbn="978-3-030-79463-7"
}
```


<!-- ACKNOWLEDGEMENTS -->
## Acknowledgements
* [Improving Sequence Tagging for Vietnamese Text Using Transformer-based Neural Models](https://arxiv.org/abs/2006.15994)
* [Unsupervised Cross-lingual Representation Learning at Scale](https://arxiv.org/pdf/1911.02116.pdf)
* [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)
