#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")
sys.path.append("./tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import test_classifier, dump_classifier_and_data
import pandas as  pd
import numpy as np

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','salary'] # You will need to use more features
features_list = ['poi','long_term_incentive', 'total_payments', 'total_stock_value',  'salary','bonus'] 

### Load the dictionary containing the dataset
data_dict = pickle.load(open("final_project_dataset.pkl", "r") )

### Task 2: Remove outliers

## Remove TOTAL not a person
data_dict.pop('TOTAL', 0)
data_dict.pop('THE TRAVEL AGENCY IN THE PARK', 0)

## Remove Lockhart, has no available values
data_dict.pop('LOCKHART, EUGENE E', 0)

### Task 3: Create new feature(s)
def computeFraction( poi_messages, all_messages ):
    """ given a number messages to/from POI (numerator) 
        and number of all messages to/from a person (denominator),
        return the fraction of messages to/from that person
        that are from/to a POI
    """

    ### you fill in this code, so that it returns either
    ###     the fraction of all messages to this person that come from POIs
    ###     or
    ###     the fraction of all messages from this person that are sent to POIs
    ### the same code can be used to compute either quantity

    ### beware of "NaN" when there is no known email address (and so
    ### no filled email features), and integer division!
    ### in case of poi_messages or all_messages having "NaN" value, return 0.

    try:
        fraction = (poi_messages + 0.) / all_messages
    except:
        fraction = 0
    return fraction

## Load DataFrame for easy engineering
df = pd.DataFrame(data_dict).T

## Add Features: c1, eu
##
## Perform KMeans cluster, use the new coordinates as a new feature
##
from sklearn.cluster import KMeans
df_drop = df.drop(['poi', 'email_address', 'to_messages', 'from_messages', 'shared_receipt_with_poi', 'other', 'from_this_person_to_poi', 'from_poi_to_this_person', 'expenses'], 1)

df_drop = df_drop.replace('NaN', 0)
cluster = KMeans(n_clusters=3, tol=.000000000001)
coord = cluster.fit_transform(df_drop.values)
c1 = coord[:,0]
c2 = coord[:,1]
eu = (coord[:,0]**2 + coord[:,1]**2)**.5
df['c1'] = c1
df['c2'] = c2
df['eu'] = eu



## Add Feature: fraction_to_poi
##
##     Intuition: POI send proportionally more mails to other poi
## 
from_poi = df.from_poi_to_this_person.values
to_messages = df.to_messages.values
from_this_person = df.from_this_person_to_poi.values
from_messages = df.from_messages
fraction_from_poi = map(computeFraction, from_poi, to_messages)
fraction_to_poi = map(computeFraction, from_this_person, from_messages)
df['fraction_from_poi'] = fraction_from_poi
df['fraction_to_poi'] = fraction_to_poi

## Add Feature: imbalance_payment_stock
##
##    Intuition: disproportionate difference between total_stock_value and total_payments indicates a fraud
## 
total_payments = df.total_payments.replace('NaN', 0)
total_stock_value = df.total_stock_value.replace('NaN', 0)
df['imbalance_payment_stock'] = abs(total_payments - total_stock_value)

## Add Feature: fraction_total_exercised_stock
##
##      Intuition: disproportionally more total stock than exercised stock could indicate fraud
##
exercised_stock_options = df.exercised_stock_options.replace('NaN', 1)
total_stock_value = df.total_stock_value.replace('NaN', 1)

df['fraction_total_exercised_stock'] = map(computeFraction, total_stock_value, exercised_stock_options)

## Add Feature: fraction_to_from_poi
##
##      Intuition: proportionally more mails to poi than from poi could indicate fraud
##
fraction_to_from_poi = map(computeFraction, fraction_to_poi, fraction_from_poi)
df['fraction_to_from_poi'] = fraction_to_from_poi



### Add all new features to the features_list
features_list += ['fraction_to_poi', 'fraction_total_exercised_stock','eu', 'c1','fraction_to_from_poi', 'imbalance_payment_stock']

# Convert all the new features into a dictionary
data_dict = df.T.to_dict()

## Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)


### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, ExtraTreesClassifier, AdaBoostClassifier, GradientBoostingClassifier
# clf = GaussianNB()    # Provided to give you a starting point. Try a varity of classifiers.
# clf = RandomForestClassifier()
# clf = ExtraTreesClassifier()
# clf = AdaBoostClassifier()


from sklearn.decomposition import RandomizedPCA, PCA
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC, SVC, NuSVC
from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import StandardScaler, MinMaxScaler, Imputer
from sklearn.feature_selection import SelectPercentile, SelectKBest, f_classif
from sklearn.tree import DecisionTreeClassifier
from sklearn.lda import LDA
from sklearn.qda import QDA
# clf = GradientBoostingClassifier()
imp = Imputer(missing_values='NaN', strategy='mean', axis=1)

clf = Pipeline([
#
#    ('standardscale', StandardScaler()),
#    ('standardscale', MinMaxScaler()),
#    ('kbest', SelectKBest(score_func=f_classif)),
#    ('pca', PCA()),
#    ('standardscale', MinMaxScaler()),
#    ('PCA', RandomizedPCA(n_components=2, whiten=False)),
#    ('SelectPercentile', SelectPercentile(percentile=.5)),
#    ('KBest', SelectKBest(score_func=f_classif, k=2)),        
#    ('imp', imp),
#    ('feature_selection', LinearSVC(C=1, penalty='l2', dual=True)),
#    ('classification', SVC(C=1, kernel='sigmoid'))
#    ('classification', RandomForestClassifier(n_estimators=100, max_depth=4, min_samples_split=2))
#    ('log_reg', LogisticRegression(C=.0011, tol=.000000000001))
    ('lda', LDA())
#    ('classification2', NuSVC(nu=.2, kernel='sigmoid', degree=2))
#    ('classification', GaussianNB())
])





### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script.
### Because of the small size of the dataset, the script uses stratified
### shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

test_classifier(clf, my_dataset, features_list)


### Dump your classifier, dataset, and features_list so 
### anyone can run/check your results.

dump_classifier_and_data(clf, my_dataset, features_list)










