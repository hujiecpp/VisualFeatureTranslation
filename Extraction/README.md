# Prerequisites

# Extraction
## SIFT

## DELF

## CroW

## MAC

## GeM

# Evaluation
After extraction, we use the official implementation to evaluate the mAP of test datasets ([Holidays](http://lear.inrialpes.fr/people/jegou/data.php#holidays), [Oxford5k](http://www.robots.ox.ac.uk/~vgg/data/oxbuildings/), [Paris6k](http://www.robots.ox.ac.uk/~vgg/data/parisbuildings/)).

## Holidays
1. Get the features for the galary data and query data.
2. Download the [Evaluation Package](https://lear.inrialpes.fr/~jegou/code/eval_holidays.tgz) from the official site and unzip it to the same path with file `test_Holidays.py`.
2. Run the brute-force retrieval for features by assigning the **feature_name**, **feature_dim**, **galary_path**, **query_path**. The **feature_name** is the type of features to be tested, the **feature_dim** is the dimension of the feature, the **galary_path** is the feature path of extracted features for the images to be retrieved, and the **query_path** is the feature path of extracted features for the query images.
> python test_Holidays.py --feature_name x --feature_dim x --galary_path x --query_path x

## Oxford5k

## Paris6k