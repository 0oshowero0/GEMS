## Genetic Meta-Structure Search for Recommendation on Heterogeneous Information Network

### 

### Introduction

This repo is an implimentation of CIKM2020 paper *Genetic Meta-Structure Search for Recommendation on Heterogeneous Information Network*.



### Dependencies

python 3.7

pytorch 1.2.0

torch geometric

numpy

pandas

h5py

tensorboardX

setproctitle

GPUtil

matplotlib





### Dataset

We use Yelp dataset provided in [HIN-Datasets-for-Recommendation-and-Network-Embedding](https://github.com/librahu/HIN-Datasets-for-Recommendation-and-Network-Embedding). You can star this dataset repo as you like, and in this repo we already integrated necessary data.

Here, we use the following node types of Yelp dataset to generate our HIN. The bolded relation is the main relation.

| Relations (A-B) | #A | #B | #A-B|
| ------ | ----------- | ----------- | ----------- |
| **User-Business (U-B)** | **16239** | **14284** | **198397** |
| User-User (U-U)   | 16239 | 16239 | 158590 |
| User-Compliment (U-O)   | 16239 | 11 | 76875 |
| Business-City (B-I)   | 14284 | 47 | 14267 |
| Business-Category (B-I)   | 14284 | 511 | 40009 |



### Usage

 0. Data Process (You do **not** need to perform this step)

     ``` bash
     cd data_process
     python trans_data_yelp.py
     python BMF.py
     ```

     trans_data_yelp.py will generate a dataset file which is needed for GEMS. We already generated this dataset file and put it under the main directory called yelp_dataset.hdf5.

     BMF.py is one of the baseline, and it's also performs as a pre-train embedding generator. The pre-train embeddings are located under MF_pretrain directory.

 1. Train the model

     ``` bash
     mkdir result_log
     mkdir result_log/yelp
     mkdir result_edges
mkdir error_genes_results
     python GEMS_yelp.py
     ```
     
     Note that you may need to change the multi-process training setup according to your server. Search #IMPORTANT in GEMS_yelp.py to locate multi-process setup.
     
     
     
1. Try to use predictor when you got some results
  
  ```bash
   python GEMS_yelp_with_predictor.py
  ```
  ​      




​     
