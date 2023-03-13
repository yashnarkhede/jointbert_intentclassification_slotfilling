# JointBERT

(Unofficial) Pytorch implementation of `JointBERT`: [BERT for Joint Intent Classification and Slot Filling](https://arxiv.org/abs/1902.10909)

- Predict `intent` and `slot` at the same time from **one BERT model** (=Joint model)
- total_loss = intent_loss + coef \* slot_loss (Change coef with `--slot_loss_coef` option)
- **If you want to use CRF layer, give `--use_crf` option**

## Dependencies

- python>=3.6
- torch==1.6.0
- transformers==3.0.2
- seqeval==0.0.12
- pytorch-crf==0.7.2

- The number of labels are based on the _train_ dataset.
- Add `UNK` for labels (For intent and slot labels which are only shown in _dev_ and _test_ dataset)
- Add `PAD` for slot label

## Training & Evaluation

```bash
$ python3 main.py --task {task_name} \
                  --model_type {model_type} \
                  --model_dir {model_dir_name} \
                  --do_train --do_eval \
                  --use_crf

# For ATIS
$ python3 main.py --task atis \
                  --model_type bert \
                  --model_dir atis_model \
                  --do_train --do_eval
# For Snips
$ python3 main.py --task snips \
                  --model_type bert \
                  --model_dir snips_model \
                  --do_train --do_eval
```

## Prediction

```bash
$ python3 predict.py --input_file {INPUT_FILE_PATH} --output_file {OUTPUT_FILE_PATH} --model_dir {SAVED_CKPT_PATH}
```
