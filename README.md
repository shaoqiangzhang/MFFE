# MFFE  (Multi-feature fusion embeddings for scHi-C)


## `***_classification` Overview

1.  Cell classification on each dataset corresponds to a directory `***_classification`, and the files under each directory follow the similar logic.

2. The subdirectory `***_Data` contains raw single-cell Hi-C contact data from different cell types.

3. The subdirectory `Feature_extraction` contains eight code files for generating eight feature sets (DBCP, ENBCP, MRBCP, NBCP, PSDCP, RBCP, SBCP, and SSDCP), and the file `methods.py` contains several functions shared by the  eight code files.

4. The subdirectory `***_features` stores all features created by Python scripts in `Feature\_extraction`.

5. The subdirectory `train_and_test` contains the code for training and testing the MFFE framework. The file `main_ablation.py` contains the main code of the MFFE framework for training on the training set and testing on the test set, and the file `method_ablation.py` contains the declarations of certain functions and classes called in `main_ablation.py`.

6. The file `combo_hg19.genomesize` contains the lengths of the 23 chromosomes in human cells, measured as the number of base pairs in each chromosome. The file `mm10.main.nochrM.chrom.sizes` contains the lengths of the 20 chromosomes in mouse cells, measured as the number of base pairs in each chromosome.
