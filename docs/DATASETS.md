## [Dataset description](https://www.nature.com/articles/s41597-020-00715-8)


DATA FORMAT

All files are stored in Nifti-1 format with 32-bit floating point data. 

Images are stored as 'volume-XX.nii.gz' where XX is the case number. All images are CT scans, under a wide variety of imaging conditions including high-dose and low-dose, with and without contrast, abdominal, neck-to-pelvis and whole-body. Many patients exhibit cancer lesions, especially in the liver, but they were not selected according to any specific disease criteria. Numeric values are in Hounsfield units.

Segmentations are stored as 'labels-XX.nii.gz', where XX is the same number as the corresponding volume file. Organs are encoded as follows:

* 0: Background (None of the following organs)
* 1: Liver
* 2: Bladder
* 3: Lungs
* 4: Kidneys
* 5: Bone
* 6: Brain

## AMOS22 
Data Info
Two databases are used in the challenge: Abdominal CT and MRI. Each data set in these two databases corresponds to a series of DICOM images belonging to a single patient. The data sets are collected retrospectively from Longgang District Central Hospital (SZ, CHINA) and Longgang District People's Hospital (SZ, CHINA). There is no connection between the data sets obtained from CT and MR databases (i.e. they are acquired from different patients and not registered).

A total of 500 CT and MRI  with annotations of 15 organs (spleen, right kidney, left kidney, gallbladder, esophagus, liver, stomach, aorta, inferior vena cava, pancreas, right adrenal gland, left adrenal gland, duodenum, bladder, prostate/uterus) are presented. Note that, some data points have missing certain organs due to physiological removal or due to parts of the body not being scanned.

Training and Test Data
The training data for both tasks can be accessed on May 1st. The training data contains complete image (nifty format) and their ground truths for the selected data sets (i.e. patients).  For the 500 CT scans of task 1, it is planned to make a 40%/20%/40% split to get 200 training cases, 100 validation cases, and 200 testing cases. 

As for task  2, the 500 CT and 100 MRI scans are planned to be split to get 200 CT + 40 MRI training cases, 100 CT + 20 MRI validation cases, and 200 CT + 40 MRI testing cases. The split of CT data will remain the same in both tasks.

In summary,  we published training data (200 CT + 40 MRI scans), and unannotated validation data (100 CT + 20 MRI scans) in the first phase of evaluation.  


