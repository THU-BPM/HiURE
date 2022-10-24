# HiURE: Hierarchical Exemplar Contrastive Learning for Unsupervised Relation Extraction

![image-20220620185455655](https://tva1.sinaimg.cn/large/e6c9d24ely1h3ewpddnqnj21ma0j8dlt.jpg)

This project provides Pytorch implementation for the paper [HiURE: Hierarchical Exemplar Contrastive Learning for Unsupervised Relation Extraction](https://arxiv.org/abs/2205.02225). 

More details about the work are in the paper.

## Quick Links
- [Get Started](#Get Started)
- [Data](#data)
- [Training](#Training)
- [Acknowledgements](#acknowledgements)
- [Contact](#Contact)

## Get Started

This implementation only supports multi-gpu, DistributedDataParallel training, which is faster and simpler; single-gpu or DataParallel training is not supported.

### Requirements

* NYT or Tacred dataset
* Python ≥ 3.6
* PyTorch ≥ 1.7
* <a href="https://github.com/facebookresearch/faiss">faiss-gpu</a>: pip install faiss-gpu

### PyTroch

The code is based on PyTorch 1.7. You can find tutorials [here](https://pytorch.org/tutorials/).

### Dependencies

The code is written in Python 3.7. Its dependencies are summarized in the file ```requirements.txt```. 

```
numpy==1.17.4
six==1.12.0
pandas==1.0.3
tqdm==4.40.0
scipy==1.4.1
bcubed==1.5
faiss==1.5.3
Pillow==8.1.0
scikit_learn==0.24.1
tensorboard_logger==0.1.0
torch==1.7.1
torchvision==0.8.2
transformers==4.2.2
```

You can install these dependencies like this:
```
pip3 install -r requirements.txt
```
Note that faiss needs to be installed as  [Requirements](#Requirements) said.


## Data

### Download

* NYT+FB: This dataset is not open, so only [sample](https://github.com/diegma/relation-autoencoder/blob/master/data-sample.txt) is provided.<br>
* TACRED: The TAC Relation Extraction Dataset ([download](https://catalog.ldc.upenn.edu/LDC2018T24))<br>

Then use the scripts from ```data_process/NYT_data_process.py``` or ```data_process/tacred_data_process.py```  to further preprocess the data. Both dataset will be processed into the [Format](#Format)

### Format
Each dataset is a folder under the ```./data``` folder:
```
./data
└── Tacred
    ├── train_sentence.json
    ├── train_label_id.json
    ├── dev_sentence.json
    ├── dev_label_id.json
    ├── test_sentence.json
    └── test_label_id.json

```


## Training
* Run the full model on TACRED dataset with default hyperparameter settings<br>

### Run parameters

```
python main_HiURE.py \
--temperature 0.2 \
--mlp \
--aug-plus \
--cos \
--use-relation-span \
--dist-url tcp://localhost:10001 \
--multiprocessing-distributed \
--world-size 1 \
--rank 0 \
{ path_to_data }
```

## Acknowledgements
This work can not be finished without the help of the following work:

* https://github.com/huggingface/transformers

* https://github.com/salesforce/PCL
* https://github.com/facebookresearch/moco


## Contact

If you have any problem about our code, feel free to contact us.
