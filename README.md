Fast organs localization on CT
==============================

### Task 1

| iou_score | f1_score  | accuracy | recall |
|:---------:|-----------|----------|--------|
|   0.93    | 0.96      | 0.99     | 0.97   |

### Task 2
| iou_score | f1_score | accuracy | recall |
|:---------:|----------|----------|--------|
|   0.84    | 0.91     |  0.98    | 0.91   |

### Task 3 
| iou_score | f1_score | accuracy | recall | channels |
|:---------:|----------|----------|--------|----------|
|   0.91    | 0.94     | 0.99     | 0.94   | 3        |
|   0.90    | 0.94     | 0.99     | 0.93   | 5        |
|   0.90    | 0.93     | 0.99     | 0.93   | 10        |



|                             Predictions                             |                           Targets                           |
|:-------------------------------------------------------------------:|:-----------------------------------------------------------:|
| ![reports/figures/predictions.png](reports/figures/predictions.png) | ![reports/figures/targets.png](reports/figures/targets.png) |

Project Organization
------------
    ├── README.md   
    ├── data
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, zipped data.
    ├── docs
    ├── models             <- Trained and serialized models
    ├── notebooks
    ├── reports   
    │   └── figures        <- Generated graphics and figures to be used in reporting
    ├── requirements.txt 
    └──  src                <- Source code for use in this project.
       │
       ├── data           <- Scripts generate dataset
       │   └── make_dataset.py
       ├── models         <- Scripts to train models and then use trained models to make predictions
       │   ├── predict_model.py
       │   └── train_model.py
       └── visualization  <- Scripts to create exploratory and results oriented visualizations
           └── visualize.py

--------