# MFFE  (Multi-feature fusion embeddings for scHi-C)


## `***_classification` Overview

1.  Cell classification on each dataset corresponds to a directory `***_classification`, and the files under each directory follow the similar logic.

2. The subdirectory `***_Data` contains raw single-cell Hi-C contact data from different cell types.

3. The subdirectory `Feature_extraction` contains eight code files for generating eight feature sets (DBCP, ENBCP, MRBCP, NBCP, PSDCP, RBCP, SBCP, and SSDCP), and the file `methods.py` contains several functions shared by the  eight code files.

4. The subdirectory `***_features` stores all features created by Python scripts in `Feature_extraction`.

5. The subdirectory `train_and_test` contains the code for training and testing the MFFE framework. The file `main_ablation.py` contains the main code of the MFFE framework for training on the training set and testing on the test set, and the file `method_ablation.py` contains the declarations of certain functions and classes called in `main_ablation.py`.

6. The file `combo_hg19.genomesize` contains the lengths of the 23 chromosomes in human cells, measured as the number of base pairs in each chromosome. The file `mm10.main.nochrM.chrom.sizes` contains the lengths of the 20 chromosomes in mouse cells, measured as the number of base pairs in each chromosome.

## Usage

### Step 1. Feature extraction

First, run the feature extraction files in the `Feature_extraction` subdirectory. Based on the raw single-cell Hi-C data stored in the folder `***_Data`, they generate the corresponding feature set for each cell in the corresponding dataset, and the generated feature sets are stored in the folder `***_features`.

### Step 2. Training and testing MFFE

Next, run `main_ablation.py` in the folder `train_and_test` to complete the training and testing of MFFE. `main_ablation.py` is a general-purpose main function for ablation experiments, in which only the `feature_config` configuration needs to be modified to control which feature sets are used.

To ensure the scientific validity and accuracy of the experimental results, to reduce the dependence on a specific dataset partition, and to evaluate the generalization ability of the model, we split the dataset into a training set and a test set, performing the split 50 times with a different random seed each time to ensure the randomness of the data splitting process. This process generates 50 pairs of training and test sets; each training set is used to train the model, and the trained model is then evaluated on the corresponding test set. The evaluation results are stored in the subfolder `***_train_test_results`.

