# Prototype-Based Density-Aware Active Domain Adaptation

This repository is the official implementation of "Prototype-Based Density-Aware Active Domain Adaptation".

## Setup Datasets

- Download [Office-31 Dataset](https://faculty.cc.gatech.edu/~judy/domainadapt/)
- Download [Office-Home Dataset](http://hemanthdv.org/OfficeHome-Dataset/)
- Download [VisDA-2017 Dataset](https://github.com/VisionLearningGroup/taskcv-2017-public/tree/master/classification)
- MiniDomainNet You need to download the DomainNet dataset first, and the MiniDomainNet's split files can be downloaded at this [google drive](https://drive.google.com/open?id=15rrLDCrzyi6ZY-1vJar3u7plgLe4COL7).

  The data folder should be structured as follows, and we have provided text index files:
    ```
    ├── data/
    │   ├── office-31/     
    |   |   ├── amazon/
    |   |   ├── dslr/
    |   |   ├── webcam/
    │   ├── office-home/
    |   |   ├── Art/
    |   |   ├── Clipart/
    |   |   ├── Product/
    |   |   ├── Real_World/
    │   ├── VisDA/
    |   |   ├── validation/
    |   |   ├── train/
    │   ├── domainnet/	
    |   |   ├── clipart/
    |   |   |—— infograph/
    |   |   ├── painting/
    |   |   |—— quickdraw/
    |   |   ├── real/	
    |   |   ├── sketch/	
    |   |——
    ```
  We have provided text index files.


## Training

- Pre-train model on the source domain by running:
```
python main.py --method SOURCE_ONLY --bs 32 --dataset <dataset_name> --source <source_domain_name> --target <target_domain_name>
```

- Train PDADA using the pre-trained model by running:
```
python main.py --method PDADA --resume <model_path> --dataset <dataset_name> --source <source_domain_name> --target <target_domain_name>
```

## Acknowledgments

This project is based on the following open-source project. We thank the authors for making the source code publicly available.

- [Transferable-Query-Selection](https://github.com/thuml/Transferable-Query-Selection)
