# pyECGdeli - ECG delineation algorithms for python

ECGdeli is originally a Matlab toolbox for filtering and processing single or multilead ECGs which was developed at the Institute of Biomedical Engineerin at Karlsruhe Institute for Technology. In this repository, I am presenting an implementation of the algorithms in python. I intended to use the same structure if possible, however, at some points, I tried to improve the MATLAB implementation. If you want to use the Code, please stick with the citation rules of ECGdeli as presented in this repository.

The main differences between ECGdeli and pyECGdeli using are:
* Every wave detection algorithm is multi-lead. You don't need to run a function "annotateECG_multi" as in ECGdeli.
* FPT tables are still used. Indexing, however, starts at 0.

Only code for QRS complex detection was published. R peak detection was evaluated using the QT database from physionet as with ECGdeli. Statistics on the errors on R peak detection with this database were:
-2.00 ± 3.85 (mean/std of errors)
2.60 ± 3.47 (mean/std of absolute errors)
2.00 ± 2.00 (median/iqr of absolute errors)


Please note the following points:
* This is work in progress, so certain algorithms are still missing at the moment. Tests of the existing algorithms will be conducted similar to those described in the ECGdeli paper.
* All algorithms must be used with ECGs as standing vectors or matrices with leads columnwise arranged (temporal dimension in lines)
* I publish the software as it is and do not guarantee proper performance. Nevertheless, I highly acknowledge feedback. Use the issues functionality in github.
