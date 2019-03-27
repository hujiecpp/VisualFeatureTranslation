This project is ongoing. (2019.3.27)

This is the project page of our paper:

"Towards Visual Feature Translation." Hu, J., Ji, R., Liu, H., Zhang, S., Deng, C., & Tian, Q. In CVPR 2019. \[[paper](https://arxiv.org/abs/1812.00573)\]

If you have any problem, please feel free to contact us.

# 1. Preparing Features

This section contains the process of collecting popular content-based image retrieval features for preparing the meta-data of our paper.

These features are extracted and evaluated in this section.

## 1.1. Evaluation
### Datasets
Datasets for evaluation:  
- [Holidays](http://lear.inrialpes.fr/people/jegou/data.php#holidays) [1]
- [Oxford5k](http://www.robots.ox.ac.uk/~vgg/data/oxbuildings/) [2]
- [Paris6k](http://www.robots.ox.ac.uk/~vgg/data/parisbuildings/) [3]

Dataset for PCA whitening and creating codebooks:
- [Google-Landmarks](https://www.kaggle.com/c/landmark-retrieval-challenge) [4]

### Measurement
We use the mean Average Precision (mAP) provided by the official site of above datasets for evaluation.

The detailed description of mAP and the code can be found in the following path of this project:
> ./Evaluation/

## 1.2. Features
Please note that all our extractions for features do not use the bounding boxes of images.

The local features (e.g., SIFT and DELF) are aggregated by the codebooks learned on 4,000 randomly picked images of Google-Landmarks dataset.

And the picked images are used for training the PCA whitening of all the features.

The features are listed bellow:

- SIFT-FV and SIFT-VLAD:

xxxxxxx
> ./Evaluation/SIFT/

- DELF-FV and DELV-VLAD:

xxxxxxx
> ./Evaluation/DELF/

- V-CroW and R-CroW:

xxxxxxx
> ./Evaluation/CroW/

- V-SPoC and R-SPoC:

xxxxxxx
> ./Evaluation/SPoC/

- V-MAC, V-rMAC and R-MAC, R-rMAC:

xxxxxxx
> ./Evaluation/MAC/

- V-GeM, V-rGeM and R-GeM, R-rGeM:

xxxxxxx
> ./Evaluation/GeM/

## 1.3. Results
The mAP (%) of collected features are as follows:

|          | Holidays | Oxford5k | Paris6k |
|   :---:  |:--------:|:--------:|:-------:|
|SIFT-FV   |61.77     |36.25     |36.91    |
|SIFT-VLAD |63.92     |40.49     |41.49    |
|DELF-FV   |83.42     |73.38     |83.06    |
|DELF-VLAD |84.61     |75.31     |82.54    |
|V-CroW    |83.17     |68.38     |79.79    |
|V-GeM     |84.57     |82.71     |86.85    |
|V-MAC     |74.18     |60.97     |72.65    |
|V-rGeM    |85.06     |82.30     |87.33    |
|V-rMAC    |83.50     |70.84     |83.54    |
|V-SPoC    |83.38     |66.43     |78.47    |
|R-CroW    |86.38     |61.73     |75.46    |
|R-GeM     |89.08     |84.47     |91.87    |
|R-MAC     |88.53     |60.82     |77.74    |
|R-rGeM    |89.32     |84.60     |91.90    |
|R-rMAC    |89.08     |68.46     |83.00    |
|R-SPoC    |86.57     |62.36     |76.75    |

# 2. Learning to Translate

> ./Translation/

# 3. Feature Relation Mining

> ./Relation/

# 4. Reference
[1] "Hamming embedding and weak geometric consistency for large scale image search." Jegou, H., Douze, M., & Schmid, C. In ECCV 2008.  
[2] "Object retrieval with large vocabularies and fast spatial matching." Philbin, J., Chum, O., Isard, M., Sivic, J. & Zisserman, A. In CVPR 2007.  
[3] "Lost in Quantization: Improving Particular Object Retrieval in Large Scale Image Databases." Philbin, J., Chum, O., Isard, M., Sivic, J. & Zisserman, A. In CVPR 2008.  
[4] "Large-scale image retrieval with attentive deep local features." Noh, H., Araujo, A., Sim, J., Weyand, T., & Han, B. In ICCV 2017.  
