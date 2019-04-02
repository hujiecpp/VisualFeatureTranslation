# Prerequisites
- matlab (for extracting and aggregating features)
- python2, python3, and g++ (for mAP evaluation)
- tensorflow (for extracting DELF)
- matconvnet (for extracting features)

# Extraction
1. Download the images of test datasets ([Holidays](http://lear.inrialpes.fr/people/jegou/data.php#holidays), [Oxford5k](http://www.robots.ox.ac.uk/~vgg/data/oxbuildings/), [Paris6k](http://www.robots.ox.ac.uk/~vgg/data/parisbuildings/)). And each dataset is seperated to `galary` for searching and `query` for querying.
2. Extract different types of features by the following steps.

## SIFT
1. Set up [VLFeat](http://www.vlfeat.org/install-matlab.html) for matlab: download [VLFeat binary package](http://www.vlfeat.org/download/vlfeat-0.9.21-bin.tar.gz) and unzip it.
2. Run `extract_sift.m` by assigning the **image_dir**, **save_dir** in the code. The **image_dir** is the dataset's abosolute path, and **save_dir** is the path of extrected features.
> matlab  
> \>\> extract_sift
3. After extracting the local features, revise the data path and save path in `aggre_sift_fv.m` and run it for aggregating SIFT by FV. Also `aggre_sift_vlad.m` by VLAD.
> \>\> aggre_sift_fv
> \>\> aggre_sift_vlad

## DELF
1. Set up [DELF](https://github.com/tensorflow/models/tree/master/research/delf).
2. Generate the text file for **list_images_path**.
> python imagelist.py -dir x
3. Run `extract_delf.py` for extracting.
> python extract_delf.py --list_images_path x.txt --output_dir x
4. Run `aggre_delf_fv.m` for aggregating DELF by FV, and `aggre_delf_vlad.m` by VLAD.
> matlab 
> \>\> aggre_delf_fv
> \>\> aggre_delf_vlad
5. Convert the files to .mat.
> python convert.py

## MAC
1. Set up [Matconvnet](http://www.vlfeat.org/matconvnet/).
2. Download the pre-trained models from the Matconvnet site: [vgg-16](http://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-16.mat) and [resnet-101](http://www.vlfeat.org/matconvnet/models/imagenet-resnet-101-dag.mat).
3. Run the extraction for V-MAC, V-rMAC, R-MAC and R-rMAC.
> matlab  
> \>\> extract_vgg_mac
> \>\> extract_vgg_rmac
> \>\> extract_resnet_mac
> \>\> extract_resnet_rmac

## CroW
1. The same as Step 1 above.
2. The same as Step 2 above.
3. Run the extraction for V-CroW and R-CroW.
> matlab  
> \>\> extract_vgg_crow
> \>\> extract_resnet_crow

## SPoC
1. The same as Step 1 above.
2. The same as Step 2 above.
3. Run the extraction for V-SPoC and R-SPoC.
> matlab  
> \>\> extract_vgg_spoc
> \>\> extract_resnet_spoc

## GeM
1. Download and set up the official implementation of [GeM](https://github.com/filipradenovic/cnnimageretrieval).
2. Run the extraction for V-GeM, V-rGeM, R-GeM and R-rGeM.
> matlab  
> \>\> extract_vgg_gem
> \>\> extract_vgg_rgem
> \>\> extract_resnet_gem
> \>\> extract_resnet_rgem

# Evaluation
After extraction, we use the official implementation to evaluate the mAP of test datasets ([Holidays](http://lear.inrialpes.fr/people/jegou/data.php#holidays), [Oxford5k](http://www.robots.ox.ac.uk/~vgg/data/oxbuildings/), [Paris6k](http://www.robots.ox.ac.uk/~vgg/data/parisbuildings/)).

## Holidays
1. Extract the features for the galary data and query data.
2. Download the [Evaluation Package](https://lear.inrialpes.fr/~jegou/code/eval_holidays.tgz) from the official site and unzip it. Then, move the `holidays_images.dat` and `holidays_map.py` to the same path with file `test_Holidays.py`.
3. Run the brute-force retrieval for features by assigning the **feature_name**, **feature_dim**, **galary_path**, **query_path**. The **feature_name** is the type of features to be tested, the **feature_dim** is the dimension of the feature, the **galary_path** is the feature path of extracted features for the images to be retrieved, and the **query_path** is the feature path of extracted features for the query images.
> python test_Holidays.py --feature_name x --feature_dim x --galary_path x --query_path x

## Oxford5k
1. Extract the features for the galary data and query data.
2. Download the [Groundtruth](http://www.robots.ox.ac.uk/~vgg/data/oxbuildings/gt_files_170407.tgz) from the official site, and unzip this file to the same path with `test_Oxford5k.py`. Then, rename the file by using `mv gt_files_170407 Oxford5k_gnd`.
3. Download [C++ code](http://www.robots.ox.ac.uk/~vgg/data/oxbuildings/compute_ap.cpp) from the official site.
4. Compile the `compute_ap.cpp` file by using `g++ -O compute_ap.cpp -o compute_ap`.
5. Run the brute-force retrieval for features by assigning the **feature_name**, **feature_dim**, **galary_path**, **query_path**. The **feature_name** is the type of features to be tested, the **feature_dim** is the dimension of the feature, the **galary_path** is the feature path of extracted features for the images to be retrieved, and the **query_path** is the feature path of extracted features for the query images.
> python test_Oxford5k.py --feature_name x --feature_dim x --galary_path x --query_path x

## Paris6k
1. Extract the features for the galary data and query data.
2. Download the [Groundtruth](http://www.robots.ox.ac.uk/~vgg/data/parisbuildings/paris_120310.tgz) from the official site, and unzip this file to the same path with `test_Paris6k.py`. Then, rename the file by using `mv paris_120310 Paris6k_gnd`.
3. Run the brute-force retrieval for features by assigning the **feature_name**, **feature_dim**, **galary_path**, **query_path**. The **feature_name** is the type of features to be tested, the **feature_dim** is the dimension of the feature, the **galary_path** is the feature path of extracted features for the images to be retrieved, and the **query_path** is the feature path of extracted features for the query images.
> python test_Paris6k.py --feature_name x --feature_dim x --galary_path x --query_path x