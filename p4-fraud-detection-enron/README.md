## Enron POI Identifier Report

by Wai Kit Tsang for the Udacity Machine Learning Course

## Introduction

In 2000, Enron was one of the largest companies in the United States. By 2002, it had collapsed into bankruptcy due to widespread corporate fraud. In the resulting Federal investigation, there was a significant amount of typically confidential information entered into public record, including tens of thousands of emails and detailed financial data for top executives. 

This particular case of fraud has been well-documented with email and financial data being publicly available. In this projec the data of 146 executives at Enron are used to identify the persons of interest in the fraud case. A person of interest (POI) is someone who was indicted for fraud, settled with the government, or testified in exchange for immunity.  This report documents the machine learning techniques used in building a POI identifier. 

## Understanding the Dataset and Question

### Data Exploration
The dataset includes 146 observations with 18 POI and 21 features. 14 features relate the financial aspects, 6 features pertained to email information and 1 boolean feature indicated whether the person was a POI. Several of these features contain missing values, but this should not hinder their usage for machine learning methods. Moreover, a lot of the data are noisy, i.e. with data points that go against the general intuition (f.e. `long_term incentive`, `bonus` or `salary` cf. figures below). 

![long_term_incentive](https://cloud.githubusercontent.com/assets/10603363/8902059/2df03714-344e-11e5-8fcc-85a70a60c523.jpg)

POI are more likely to have higher incentive, since the median for POI is higher than the median of non-POI. Still, there is one non-POI observation with a higher long_term_incentive than all the other POI. Conversely, there is one POI with a lower long_term_incentive than the median of the non-POI.

![bonus](https://cloud.githubusercontent.com/assets/10603363/8901542/0e8556c4-344a-11e5-8c2c-b556f6358e78.jpeg)

Although one can be a POI, this does not guarantee over the top bonuses since there is one POI who received barely any bonus. 

![salary_bonus](https://cloud.githubusercontent.com/assets/10603363/8902131/bfc9d4ec-344e-11e5-9a73-480126e26132.jpeg)

Finally, relationships between salary, bonus or other features and involvement in the fraud are not that straightforward to interpret. There is even one POI who receives almost no bonus and salary. The fruits of his fraud involvement will surely be reaped elsewhere. 


### Outlier Investigation
Outliers and nonsense data are removed.
 
* `TOTAL`: Outlier and a result of transcribing the summary sheet.
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

![kmeanscluster](https://cloud.githubusercontent.com/assets/10603363/8901550/216ddea0-344a-11e5-80a8-6adc69eb8c8b.jpeg)

### Intelligently select features


