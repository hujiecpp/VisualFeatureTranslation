# Prerequisites
- python2
- python3
- g++
- matlab

# Extraction
## SIFT

## DELF

## CroW

## MAC

## GeM

# Evaluation
After extraction, we use the official implementation to evaluate the mAP of test datasets ([Holidays](http://lear.inrialpes.fr/people/jegou/data.php#holidays), [Oxford5k](http://www.robots.ox.ac.uk/~vgg/data/oxbuildings/), [Paris6k](http://www.robots.ox.ac.uk/~vgg/data/parisbuildings/)).

## Holidays
1. Extract the features for the galary data and query data.
2. Download the [Evaluation Package](https://lear.inrialpes.fr/~jegou/code/eval_holidays.tgz) from the official site and unzip it to the same path with file `test_Holidays.py`.
3. Run the brute-force retrieval for features by assigning the **feature_name**, **feature_dim**, **galary_path**, **query_path**. The **feature_name** is the type of features to be tested, the **feature_dim** is the dimension of the feature, the **galary_path** is the feature path of extracted features for the images to be retrieved, and the **query_path** is the feature path of extracted features for the query images.
> python test_Holidays.py --feature_name x --feature_dim x --galary_path x --query_path x

## Oxford5k
1. Extract the features for the galary data and query data.
2. Download and unzip the [Groundtruth](http://www.robots.ox.ac.uk/~vgg/data/oxbuildings/gt_files_170407.tgz) to the same path with file `test_Oxford5k.py`. Then, rename the file by using `mv gt_files_170407 Oxford5k_gnd`.
3. Download [C++ code](http://www.robots.ox.ac.uk/~vgg/data/oxbuildings/compute_ap.cpp) from the official site.
4. Compile the `compute_ap.cpp` file by using `g++ -O compute_ap.cpp -o compute_ap`.
5. Run the brute-force retrieval for features by assigning the **feature_name**, **feature_dim**, **galary_path**, **query_path**. The **feature_name** is the type of features to be tested, the **feature_dim** is the dimension of the feature, the **galary_path** is the feature path of extracted features for the images to be retrieved, and the **query_path** is the feature path of extracted features for the query images.
> python test_Oxford5k.py --feature_name x --feature_dim x --galary_path x --query_path x

## Paris6k