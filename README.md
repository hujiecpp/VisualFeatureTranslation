This project is ongoing. (2019.3.27)

This is the project page of our paper:

"Towards Visual Feature Translation." Hu, J., Ji, R., Liu, H., Zhang, S., Deng, C., & Tian, Q. *In CVPR 2019.* \[[paper](https://arxiv.org/abs/1812.00573)\]

If you have any problem, please feel free to contact us.

# 1. Preparing Features

This section contains the process of collecting popular content-based image retrieval features for preparing the meta-data of our paper.

The extracted features are evaluated in this section.

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

The details can be found in:
> ./Evaluation/

## 1.2. Features
Please note that our extractions for images do not use the bounding boxes of the objects.

The local features (e.g., SIFT and DELF) are aggregated by the codebooks learned on 4,000 randomly picked images of Google-Landmarks dataset.

And the features of picked images are used to train the PCA whitening for all features of other images.

The features are listed bellow:

- **SIFT-FV** and **SIFT-VLAD**: The Scale Invariant Feature Transform (SIFT) [5] features are extracted and then aggregate by Fisher Vector (FV) [6] and Vector of Locally Aggregated Descriptors (VLAD) [7]. The details can be found in:
> ./Evaluation/SIFT/
- **DELF-FV** and **DELV-VLAD**: The DEep Local Features (DELF) [8] are extracted and then aggregate also by FV and VLAD. The details can be found in:
> ./Evaluation/DELF/
- **V-CroW** and **R-CroW**: 
> ./Evaluation/CroW/

- **V-SPoC** and **R-SPoC**:
> ./Evaluation/SPoC/

- **V-MAC**, **V-rMAC** and **R-MAC**, **R-rMAC**:
> ./Evaluation/MAC/

- **V-GeM**, **V-rGeM** and **R-GeM**, **R-rGeM**:
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
[1] "Hamming embedding and weak geometric consistency for large scale image search." Jegou, H., Douze, M., & Schmid, C. *In ECCV 2008.*  
[2] "Object retrieval with large vocabularies and fast spatial matching." Philbin, J., Chum, O., Isard, M., Sivic, J. & Zisserman, A. *In CVPR 2007.*  
[3] "Lost in Quantization: Improving Particular Object Retrieval in Large Scale Image Databases." Philbin, J., Chum, O., Isard, M., Sivic, J. & Zisserman, A. *In CVPR 2008.*  
[4] "Large-scale image retrieval with attentive deep local features." Noh, H., Araujo, A., Sim, J., Weyand, T., & Han, B. *In ICCV 2017.*  
[5] "Distinctive image features from scale-invariant keypoints." Lowe, D. G. *IJCV 2004.*  
[6] "Large-scale image retrieval with compressed fisher vectors." Perronnin, F., Liu, Y., Sánchez, J., & Poirier, H. *In CVPR 2010.*  
[7] "Aggregating local descriptors into a compact image representation." Jégou, H., Douze, M., Schmid, C., & Pérez, P. *In CVPR 2010.*  
[8] "Large-scale image retrieval with attentive deep local features." Noh, H., Araujo, A., Sim, J., Weyand, T., & Han, B. *In ICCV 2017.*  
