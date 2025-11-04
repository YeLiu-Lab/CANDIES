# Cross-modal Denoising and Integration of Spatial Multi-omics data with CANDIES

## Abstract
Spatial multi-omics data offer a powerful framework for integrating diverse molecular profiles while maintaining the spatial organization of cells. However, inherent variations in data quality and noise levels across different modalities pose significant challenges to accurate integration and analysis. In this paper, we introduce CANDIES, which leverages a conditional diffusion model and contrastive learning to effectively denoise spatial multi-omics data while generating a unified and comprehensive representation. CANDIES shows superior performance on various downstream tasks, including denoising, spatial domain identification, pseudo-Spatiotemporal Map (pSM) construction, and trait domain association map (gsMap) generation. We conducted extensive evaluations on diverse synthetic and real datasets, including spatial CITE-seq data from human skin biopsy tissue, MISAR-seq data from the mouse brain, spatial ATAC-RNA-seq data from the mouse embryo and 10$\times$ visium data from human lymph nodes. Through gsMap analysis, we identified significant correlations between traits and tissues, notably the heritability enrichment of psychiatric and behavioral traits in the dorsal pallium, thalamus, and hindbrain of the mouse brain. These findings demonstrate that CANDIES effectively generates unified representations that accurately capture critical biological insights.

![image](https://github.com/zouwanpeng/CANDIES/blob/main/CANDIES.png)

## Requirements
You'll need to install the following packages in order to run the codes.
* ﻿anndata==0.11.0
* einops==0.8.0
* esda==2.7.0
* networkx==2.6
* numpy==1.26.4
* rpy2==3.5.16
* scanpy==1.10.3
* scikit-learn==1.5.2
* squidpy==1.6.2
* torch==2.5.1
* torch-geometric==2.6.1
* torchvision==0.20.1
* R==4.0.3

## Tutorial
For the step-by-step tutorial, please refer to: https://github.com/YeLiu-Lab/CANDIES/tree/main/CANDIES/tutorial

## Benchmarking methods
In the paper, we compared CANDIES with 7 unimodal clustering methods, 3 non-spatial multi-omics integration approaches, and 4 spatial multi-omics integration techniques.

## Downstream analyses
CANDIES shows superior performance on various downstream tasks, including denoising, spatial domain identification, pseudo-Spatiotemporal Map (pSM) construction, and trait domain association map (gsmap) generation.
