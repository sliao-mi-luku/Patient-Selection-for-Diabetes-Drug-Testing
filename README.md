# Patient Selection from EHR Data for Diabetes Drug Testing
Use deep learning to make decisions on clinical trials from EHR data

---
*Last updated: 12/24/2021*

[![EHR-frontpage-photo.png](https://i.postimg.cc/NMwjkLhz/EHR-frontpage-photo.png)](https://postimg.cc/PL2HtX5W)
<p align="center">
    Patient selection with TensorFlow Feature Column API and probability layers
</p>

# Project Summary

1. This project trains a deep learning model to select patients for clinical trials
2. Clean, extract, and process EHR datasets, with exploratory data analysis
3. Use TensorFlow Dataset API to build EHR datasets aggregated to the patient level
4. Performs feature engineering by TensorFlow Feature Column API
5. Train a deep learning model to predict the hospitalization duration
6. Use probability layers to generate the confidence level of the prediction
7. Analyze the biases of the model across demographic groups representing the data


# Dataset

The dataset of this project is from the [UC Irvine Diabetes Dataset](https://archive.ics.uci.edu/ml/datasets/Diabetes+130-US+hospitals+for+years+1999-2008).

# Model

We use a deep neural network to predict the hospitalized time. The input layer of the model is TF DenseFeatures, followed by 2 dense layers of 256 and 128 hidden units. We attach TF probabilities layers to the end of the model so that the prediction of the model can have its confidence intervals. With TF, we can interpret the prediction of the model.

We use RMSProp as the optimizer and train the model for 10 epochs.

[![EHR-training.png](https://i.postimg.cc/T2BBzTD0/EHR-training.png)](https://postimg.cc/5jqPvdpQ)
<p align="center">
    The model is trained for 10 epochs
</p>

The TF probability layers give us the mean and standard deviation of the prediction of the hospitalization time.

[![EHR-predictions.png](https://i.postimg.cc/2S8c3xVH/EHR-predictions.png)](https://postimg.cc/KRwrHB7L)
<p align="center">
    Model prediction, actual value, and mean and std of the population where the prediction is drawn from
</p>

# Model Evaluation

**Precision and Recall**

Precision is the ratio of the number of true positives to the sum of the true positives and false positives. Recall is the ratio of the number of true positives to the sum of the true positives and false negatives. There's a tradeoff between them. If we increase the precision (we say no to the data that we're less confident in classifying as positive), we will classify more true positives as negative and as a result decreasing the recall. On the other hand, if we increase the recall (by saying yes to the data that we're less confident in classifying it as negative), we will classify more true negative data as positive and as a result decreasing the precision.

The model achieves **0.8505 precision** and **0.1901 recall** (F1-score=0.6243). The model sacrifices the recall so that a high precision can be achieved. The precision, defined as `TP/(TP+FP)`, is more crucial for patients selection because we don't want to include patients who will not spend enough time in the hospital into the drug testing.

For future improvement, I'd like to try to increase the recall without sacrificing the precision too much. I'll tune the parameters of the model such as the number of hidden layers and hidden units. Use some regularization method such as dropout layers, to see how the model performs. I can also try different feature selections.

**Model Biases**

We see that the false positive rate is fair in gender but not in race. The false negative rate is fair in both gender and race.

From the disparity figures below we see that compared to Caucasian (the reference group), all other races have more false positive rate. On the other hand, we don't see this in gender. Gender doesn't not have much influence on the false positive rate.

[![EHR-biases.png](https://i.postimg.cc/662Mv36X/EHR-biases.png)](https://postimg.cc/DmhQV7bj)
<p align="center">
    Analysis of biases of the model
</p>


# References

1. UCI Diabetes Dataset https://archive.ics.uci.edu/ml/datasets/Diabetes+130-US+hospitals+for+years+1999-2008
2. https://github.com/udacity/nd320-c1-emr-data-starter/tree/master/project
3. https://github.com/dssg/aequitas
