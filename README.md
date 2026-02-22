# SIMPLE

A multi-slice spatial omics integration method for depicting tissue complexity via joint intra- and inter-slice learning

## Overview

![Overview of SIMPLE](./SIMPLE.png)

**a.** The input data contains gene expression matrices and spatial coordinates of multiple spatial transcriptomic datasets, and then the spatial neighborhood graph was constructed based on the coordinates. **b.** SIMPLE takes preprocessed gene expression matrix and spatial neighborhood graph as input and learns consistent spot embeddings from two perspectives: intra-slice and inter-slice. For intra-slice learning, SIMPLE uses a LightGCN-inspired network and performs augmentation directly at the representation level by adding uniform random noise to the original embeddings, which enables SIMPLE to adapt to different types of spatial transcriptomics datasets. For inter-slice learning, SIMPLE adopts both global (Optimal Transport) and local (MNN triplet pairs) strategies to remove batch effect and capture consistent biological information across different slices.

## Requirements

We recommend that users install the following packages to run spCLUE.

- python==3.8.0
- torch==1.13.1
- numpy==1.23.5
- scanpy==1.9.3
- anndata==0.8.0
- rpy2==3.4.1
- pandas==1.5.3
- scipy==1.10.0
- scikit-learn==1.2.2
- tqdm==4.64.1
- matplotlib==3.7.0
- seaborn==0.12.2
- jupyter==1.0.0
- R==4.2.0
- mclust==6.0.0

You can install SIMPLE with **anaconda** using the following commands:

```shell
conda create -n SIMPLE_env python=3.8
conda activate SIMPLE_env
pip install -r requirements.txt
```

## Tutorial

Please find examples of SIMPLE applications in the tutorials folder, where jupyter notebooks are provided.

**NOTE:** Please update the data paths before running the code.

## Datasets

The example spatial transcriptomics datasets can be downloaded with the links below.

The human DLPFC dataset is available at the spatialLIBD package (http://spatial.libd.org/spatialLIBD).

The mouse hypothalamus dataset by MERFISH and the mouse medial prefrontal cortex dataset by STARmap are available at https://github.com/zhengli09/BASS-Analysis/tree/master/data. 

The spatial proteomics datasets of mouse spleen and mouse thymus are available at https://zenodo.org/records/10362607. 
