# MVG-LGA: Multi-view graph neural network based on attention mechanism for uncovering lncRNA-related protein coding genes
> If you have any questions in using the program, please feel free to contact us by email: meihonggao@mail.nwpu.edu.cn.

## Requirements
MVG-LGA is tested in the conda environment. It is recommended that users test the program in this way. Of course, they can also execute the program in a familiar environment. Note: Before using MVG-LGA, users first need to prepare the following softwares in the execution environment：
  * Python 3.6.2
  * PyTrorch 1.9.1
  * NumPy 1.19.2
  * Scipy 1.5.2
  * scikit-learn 0.24.1

## Code
This directory stores the python code of the model
  * main_gpu.py
  >It is used to compute the performance of the model for lncRNA-PCG association prediction
  * models.py
  >It is used to define the model.
  * layers.py
  >It is used to define GAT and GCN layer.
  * utils.py
  >It is used to define the functions that need to be used in the model.

## Usage
Note: Go to the /MVG-LGA/Code/ directory before using this model.
  * Please run the following python command：```python main_cpu.py```
  
## Datasets
This directory stores the datasets used by the model
### Dataset1: NPInter dataset
  * lnc_name.txt
  > Names of lncRNAs in lnc_feat.csv
  * lnc_omic_feat.csv
  > Multi-omics features of lncRNAs
  * lnc_pcg_net.csv
  > LncRNA-PCG associations in which lncRNAs and PCGs have multi-omics features
  * lnc_seq_feat.csv
  > Sequence features of lncRNAs
  * pcg_name.txt
  > Names of protein-coding genes in pcg_feat.csv
  * pcg_omic_feat.csv
  > Multi-omics features of protein-coding genes
  * pcg_seq_feat.csv
  > Sequence features of PCGs
### Dataset2: LncTarD dataset
  * lnc_name.txt
  > Names of lncRNAs in lnc_feat.csv
  * lnc_omic_feat.csv
  > Multi-omics features of lncRNAs
  * lnc_pcg_net.csv
  > LncRNA-PCG associations in which lncRNAs and PCGs have multi-omics features
  * lnc_seq_feat.csv
  > Sequence features of lncRNAs
  * pcg_name.txt
  > Names of protein-coding genes in pcg_feat.csv
  * pcg_omic_feat.csv
  > Multi-omics features of protein-coding genes
  * pcg_seq_feat.csv
  > Sequence features of PCGs
### Dataset3: LncRNA2Target dataset
  * lnc_name.txt
  > Names of lncRNAs in lnc_feat.csv
  * lnc_omic_feat.csv
  > Multi-omics features of lncRNAs
  * lnc_pcg_net.csv
  > LncRNA-PCG associations in which lncRNAs and PCGs have multi-omics features
  * lnc_seq_feat.csv
  > Sequence features of lncRNAs
  * pcg_name.txt
  > Names of protein-coding genes in pcg_feat.csv
  * pcg_omic_feat.csv
  > Multi-omics features of protein-coding genes
  * pcg_seq_feat.csv
  > Sequence features of PCGs


