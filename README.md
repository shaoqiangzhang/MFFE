# MFFE  (Multi-feature fusion embeddings for scHi-C)


## “***_classification” Overview

1.  Cell classification on each dataset corresponds to a directory "***_classification", and the files under each directory follow the similar logic.

2. The subdirectory "***_Data" contains raw single-cell Hi-C contact data from different cell types.

3. The subdirectory "Feature_extraction" contains eight code files for generating eight feature sets (DBCP, ENBCP, MRBCP, NBCP, PSDCP, RBCP, SBCP, and SSDCP), and the file "methods.py" contains several functions shared by the  eight code files.

4. The subdirectory "***_features" stores all features created by Python scripts in "Feature\_extraction".
