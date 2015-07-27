## Enron POI Identifier Report

by Wai Kit Tsang for the Udacity Machine Learning Course

## Introduction

In 2000, Enron was one of the largest companies in the United States. By 2002, it had collapsed into bankruptcy due to widespread corporate fraud. In the resulting Federal investigation, there was a significant amount of typically confidential information entered into public record, including tens of thousands of emails and detailed financial data for top executives. 

This particular case of fraud has been well-documented with email and financial data being publicly available. In this projec the data of 146 executives at Enron are used to identify the persons of interest in the fraud case. A person of interest (POI) is someone who was indicted for fraud, settled with the government, or testified in exchange for immunity.  This report documents the machine learning techniques used in building a POI identifier. 

## Understanding the Dataset and Question

### Data Exploration
The dataset includes 146 observations with 18 POI and 21 features. 14 features relate the financial aspects, 6 features pertained to email information and 1 boolean feature indicated whether the person was a POI. Several of these features contain missing values, but this should not hinder their usage for machine learning methods. Moreover, a lot of the data are noisy, i.e. there are data points present that do not follow the general patterns in the data (f.e. `long_term incentive`, `bonus` or `salary` cf. figures below). 

![long_term_incentive](https://cloud.githubusercontent.com/assets/10603363/8902226/9c6728fa-344f-11e5-93ec-ef6a3b93b6d1.jpg)

POI are generally more likely to have a higher long term incentive, since the median for POI is higher than the median of non-POI. Still, there is one non-POI observation with a higher long_term_incentive than all the other POI. Conversely, there is one POI with a lower long_term_incentive than the median of the non-POI.

![bonus](https://cloud.githubusercontent.com/assets/10603363/8902486/4d5bd592-3451-11e5-9a19-d2a955d5ab30.jpg)

Although one can be a POI, this does not guarantee over the top bonuses since there is one POI who received barely any bonus. 

![salary_bonus](https://cloud.githubusercontent.com/assets/10603363/8902219/8dd53fb6-344f-11e5-9bdd-baac5c62ab56.jpeg)

Finally, relationships between salary, bonus or other features and involvement in the fraud are not that straightforward to interpret. There is even one POI who receives almost no bonus and salary. The fruits of his fraud involvement will surely be reaped elsewhere. 


### Outlier Investigation
Outliers and nonsense data are removed. Observations removed are:
 
* `TOTAL`: Outlier and a mistake during transcription of summary sheet.
* `LOCKHART EUGENE E`: Observation has only NaN values.
* `THE TRAVEL AGENCY IN THE PARK`: Not a person.

## Optimize Feature Selection/Engineering

### Create new features

* `fraction_to_poi`: the fraction of mails `from_this_person_poi` divided by the number of messages to this person (`to_messages`). Intuition behind this feature is that POI send proportionally more mails to other poi. 

* `imbalance_payment_stock`: The difference between `total_stock_value` and `total_payments`. Disproportionate difference between total_stock_value and total_payments might indicate fraud. Some poi fraud by manipulating their payment other by influencing the value of their stock.

* `fraction_total_exercised_stock`: fraction of total_stock_value over exercised_stock_options. POI possibly have a disproportionately more stock than their exercised stock options.

* `fraction_to_from_poi`: fraction of fraction_to_poi and fraction_from_poi. Some people generally communicate more with poi independent of their involvement in fraud. This feature can compensate for this information.

* `c1`: KMeans clustering (n=2) transforms the original features to a 2 dimensional feature space where the observations can be distinguished via the distance to the cluster means. `c1` is the first variable in the new feature space for all observations. 

* `eu`: After KMeans clustering the Euclidean distance to one of the cluster means is calculated. By clustering the observations it can be seen that POI and non-POI are more easily distinguished (cf. figure below).

![kmeanscluster]
(https://cloud.githubusercontent.com/assets/10603363/8902455/22110182-3451-11e5-81ab-ee731b796556.jpeg)

### Intelligently select features

Features are selected by hand by looking at the distribution and the effect on the accuracy of a simple model, the features were progressively added. In addition to the six newly engineered features, five of the original features were selected. By exploring the distributions in [figure](./figure), it is shown that there is a lot of noise. The SelectKBest-method was not used, since it only performs univariate feature selection, i.e. by looking within the variable for the best discrimination between the POIs and non-POIs. A strong assumption of this method is that the features are independent of each other. Extra information, however, is available namely that the total_stock_value is the sum of the `Exercised_stock_options`, `Restricted_stock`and `Resticted_stock_deferred`, while the `total_payment` is the sum of the `salary`, `bonus`, `long_term_incentive`, `deferred_income`, `deferral_payments`, `loan_advances`, `other`, `expenses` and `director fees`. 

* `long_term_incentive`

* `total_payments`

* `total_stock_value`

* `salary`

* `bonus`

### Properly Scale features

Features are not scaled since LDA (the final selected algorithm) performs a singular value decomposition which automatically looks for the most significant eigenvectors. 

## Pick and Tune an Algorithm

### Pick an algorithm

The selection of the algorithm is based on their performance with the default parameters. The algorithms were tuned but could only marginally improve their baseline-performance by tuning parameters. 

 Algorithm| Accuracy | Precision | Recall 
----------| ---------|-----------|--------
Naive Bayes | 0.862 | 0.478     | 0.409
RandomForest | 0.864 | 0.477     | 0.1795
GradientBoosting | 0.864 | 0.484     | 0.352
AdaBoost | 0.843 | 0.371     | 0.252
Logistic Regression | 0.865 | 0.485 | 0.232
Linear SVC | 0.790 | 0.190 | 0.176 | 0.182
Linear Discriminant Analysis | 0.889 | 0.615 | 0.441 
Quadratic Discriminant Analysis | 0.882 | 0.591 | 0.370

As a final choice **Linear Discriminant Analysis** was selected.

### Tune the algorithm

For RandomForest, GradientBoosting and AdaBoost the following parameters were tuned:
* n_estimators
* max_depth
* min_samples_split

The improvement on the algorithm did not surpass the score of Linear Discriminant Analysis (LDA). The tolerance was tuned for Linear SVC, Linear Discriminant Analysis and Quadratic Discriminant Analysis. Improvement was really small. For LDA the solver could also be tuned but the other options did not improve the final precision or recall. 

## Validate and Evaluate

### Usage of Evaluation Metrics
Precision and recall are used to evaluate algorithm performance. 

Precision articulates the number of true positives over the number of true positives plus the number of false positives:

![precision](https://cloud.githubusercontent.com/assets/10603363/8906883/7ae80d84-3472-11e5-9aea-ede8a37435ea.png)

Basically precision indicates how 'pure' the identifier is. In the case of the selected When the LDA classifies an unidentified observation as positive, there is approx. 61,5 % chance that it is correct

Recall is defined as the number of true positives over the number of true positives plus the number of false negatives:

![recall](https://cloud.githubusercontent.com/assets/10603363/8906889/84b81e4e-3472-11e5-9fbd-4b0ef8487561.png)

Recall conveys how many true POIs remain undetected. The closer your recall to 1, the more POIs are identified, while a recall closer to zero allow a lot of POI to stay under the radar of your identifier. 

Depending on the wishes of the client, the precision can be increased and the recall decreased, on the condition that the **cost of investigation** is really high and that the cost for investigating a POI is unreasonably higher than examining other non-POI. 

Conversely, it can also be decided to decrease the precision and increase the recall, when the **cost of investigation** is really low and finding more POI has priority. 

### Validation Strategy

In order to increase the robustness of the training and the testing cross-validation is performed. 

## Conclusion

The Final POI-identifier is based on **Linear Discriminant Analysis** which has:
* an accuracy of 0.889
* a precision of 0.615 
* a recall of 0.441. 

A balance has been struck between precision and recall. Possibly these can be further increased by looking into mail contents. 







