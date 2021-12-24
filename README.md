# Patient-Selection-for-Diabetes-Drug-Testing
Use deep learning to make decisions on clinical trials from EHR data

---
*Last updated: 12/24/2021*

# Project Summary



# Dataset

The dataset of this project is from the [UC Irvine Diabetes Dataset](https://archive.ics.uci.edu/ml/datasets/Diabetes+130-US+hospitals+for+years+1999-2008). Originally the


# Model

We use a deep neural network to predict the hospitalized time. The input layer of the model is TF DenseFeatures, followed by 2 dense layers of 256 and 128 hidden units. We attach TF probabilities layers to the end of the model so that the prediction of the model can have its confidence intervals. With TF, we can interpret the prediction of the model.


We use RMSProp as the optimizer and train the model for 10 epochs.

[![EHR-training.png](https://i.postimg.cc/T2BBzTD0/EHR-training.png)](https://postimg.cc/5jqPvdpQ)
<p align="center">
    The model is trained for 10 epochs
</p>

The TF probability layers give us the mean and standard deviation of the prediction of the hospitalization time.


# Evaluation

## Precision and Recall

The model achieves **0.8505 precision** and **0.1901 recall** (F1-score=0.6243). The model sacrifices the recall so that a high precision can be achieved. The precision, defined as `TP/(TP+FP)`, is more crucial for patients selection because we don't want to include patients who will not spend enough time in the hospital into the drug testing.

## Model Biases

We need to know how does the model perform




# References
