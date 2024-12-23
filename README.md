# AMIA Anomaly Detection

This is repository for [AMIA Public Challenge 2024](https://www.kaggle.com/competitions/amia-public-challenge-2024/overview)

Task desciption from Kaggle: 
`In this competition, we are classifying common thoracic lung diseases and localizing critical findings. This is an object detection and classification problem.`

### 0. Data Description

Kaggle dataset have the following structure

```
.
|-- train
|   |-- train
|   |   |-- {uuid}.png
|   |   |-- ...
|-- test
|   |-- test
|   |   |-- {uuid}.png
|   |   |-- ...
|-- img_size.csv
|-- sample_submission.csv
|-- test.csv
|-- train.csv
```

Train folder have 8573 png images.
Test folder have 6427 png images.

For training and testing we have only images from train folder.