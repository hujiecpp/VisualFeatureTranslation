# Prerequisites
- python3
- pytorch 0.3

# Training
We seperate the translations into groups, run the script for corresponding training.

For example:
> ./run_0_0.sh

# Test
For calculating the translation mAP results, run:
> python test_mAP.py

For calculating the Undirected Affinity Measurement, run:
> python test_Dis.py