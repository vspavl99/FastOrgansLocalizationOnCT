# AMOS22 

Two databases are used in the challenge: Abdominal CT and MRI. Each data set in these two databases corresponds 
to a series of DICOM images belonging to a single patient. The data sets are collected retrospectively 
from Longgang District Central Hospital (SZ, CHINA) and Longgang District People's Hospital (SZ, CHINA).
There is no connection between the data sets obtained from CT and MR databases (i.e. they are acquired from different
patients and not registered).

A total of 500 CT and MRI  with annotations of 15 organs (spleen, right kidney, left kidney, gallbladder, esophagus, 
liver, stomach, aorta, inferior vena cava, pancreas, right adrenal gland, left adrenal gland, duodenum, bladder, 
prostate/uterus) are presented. Note that, some data points have missing certain organs due to physiological removal or 
due to parts of the body not being scanned.

Training and Test Data
The training data for both tasks can be accessed on May 1st. The training data contains complete image 
(nifty format) and their ground truths for the selected data sets (i.e. patients).  For the 500 CT scans of task 1, 
it is planned to make a 40%/20%/40% split to get 200 training cases, 100 validation cases, and 200 testing cases. 

As for task  2, the 500 CT and 100 MRI scans are planned to be split to get 200 CT + 40 MRI training cases,
100 CT + 20 MRI validation cases, and 200 CT + 40 MRI testing cases. The split of CT data will remain the same 
in both tasks.

In summary,  we published training data (200 CT + 40 MRI scans), and unannotated validation data
(100 CT + 20 MRI scans) in the first phase of evaluation.  


