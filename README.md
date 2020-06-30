# Identifying Fraud From Enron

### About
In 2000, Enron was one of the largest companies in the United States. By 2002, it had collapsed into bankruptcy due to widespread corporate fraud. In the resulting Federal investigation, a significant amount of typically confidential information entered into the public record, including tens of thousands of emails and detailed financial data for top executives.

Using Machine Learning techniques this project will predict and spot culpable persons of the Enron scandal.

### Tech Stack
The project was built with Python 2.7 and scikit-learn library

### Data Exploration
The main goal of this project is to use both financial and email data from Enron to build a predictive model that could potentially identify a "person of interest" (POI),
i.e. Enron employees who may have committed fraud, based on the aforementioned public data.

### Data Structure
As part of the preprocessing for this project, the email and financial data from Enron has been combined into a dictionary, where each key-value pair corresponds to one person.

The dictionary key is the person's name, and the value is another dictionary, which contains the names of all the features and their values for that person.

Additionally, the features always fall into three major types:
1) Financial features (all units in US dollars):
```
salary
deferral_payments
total_payments
loan_advances
bonus
restricted_stock_deferred
deferred_income
total_stock_value
expenses
exercised_stock_options
other
long_term_incentive
restricted_stock
director_fees
```
2) Email features (units are number of emails messages
```
to_messages
email_address
from_poi_to_this_person
from_messages
from_this_person_to_poi
shared_receipt_with_poi
```

3) POI labels
```
poi
```

An Example:
```
"SKILLING JEFFREY K":
  {'salary': 1111258,
  'to_messages': 3627,
  'deferral_payments': 'NaN',
  'total_payments': 8682716,
  'exercised_stock_options': 19250000,
  'bonus': 5600000,
  'restricted_stock': 6843672,
  'shared_receipt_with_poi': 2042,
  'restricted_stock_deferred': 'NaN',
  'total_stock_value': 26093672,
  'expenses': 29336,
  'loan_advances': 'NaN',
  'from_messages': 108,
  'other': 22122,
  'from_this_person_to_poi': 30,
  'poi': True,
  'director_fees': 'NaN',
  'deferred_income': 'NaN',
  'long_term_incentive': 1920000,
  'email_address': 'jeff.skilling@enron.com',
  'from_poi_to_this_person': 88}
  ```
  
### NaNs
Despite overall having really valuable information in order to identify POIs, the data set also contained a lot of missing values — NaN. Here's a table showing the amount of NaN values per feature.
The data definitely has a lot of NaNs, the most concerning features are Loan advances, Director fees and Restricted stock deferred, which have more than 85% of their values missing.
 
### Outliers
The outlier explorations starts by plotting two of the most telling features when it comes to uncover the relation between the data and POIs: salary and bonus.
 
The plot clearly shows an outlier in the top right of the plot. Ordering the list by salary, the outlier is called TOTAL, which represents the sum of all the salaries, as shown in the insider-pay.pdf, therefore cannot be considered a person.

After removing TOTAL here's the same plot again.

It looks like there still are four additional points with higher salary and bonus, in a range that could potentially consider them as outliers.

After closer inspection, and despite not all of them were POIs, the rest of their data seemed consistent across the board and all of them looked like valid and meaningful data points.

### Incomplete Data
Another potential source of outliers are the ones that don't add meaningful information to the mix, such as persons with little or no relevant information at all.

In order to spot these data points, the get_incompletes() function returns a list of the names with no feature data above a certain threshold.

With get_incompletes() set at 90%, which means that the persons returned by the function have only less than 10% of the data completed, it returns this list.

```
['WHALEY DAVID A',
 'WROBEL BRUCE',
 'LOCKHART EUGENE E',
 'THE TRAVEL AGENCY IN THE PARK',
 'GRAMM WENDY L']
```

After inspecting closely each person one by one, there's no meaningful information we can derive from these persons and on top of that, none of each is a POI, therefore, they will be removed from the data set.

### Selected Features
The three selected algorithms are:

AdaBoost
Random Forest
SVC

AdaBoost
```
Accuracy: 0.83193
Precision: 0.38098
Recall: 0.28250
```

RandomForest
```
Accuracy: 0.84943
Precision: 0.41290
Recall: 0.12800
```

SVC
```
Accuracy: 0.49636
Precision: 0.14762
Recall: 0.52900
```  

Both AdaBoost and RandomForest got a really good Accuracy out of the box. They also got similar values for Precision, around 40%, but both felt short when it comes to Recall, under 30%.

This means that out of all the items that are truly positive, i.e. POI, how many were correctly classified as positive. Or simply put, how many positive items were 'recalled' from the data set.

### Feature Pre Selection

Going back to the table that displayed the amount of NaNs per feature, is clear that there are no features that have information for all the employees in the data set. Despite, up to five features have more than 60% missing values. If that was not enough, restricted_stock_deferred, director_fees, loan_advances and deferral_payments features are missing for more than 50% of the POI segment.

As a result, these four features and restricted_stock_deferred — with over 85% of NaN, will be omitted from the selection process.

Additionally, the email_address feature will also be left out since it is text based, and hardly provides any predictive value.

These are the updated set of features, removing the ones with higher rate of incompleteness, its respective list can be found at features_db.py

Financial features
```
salary
total_payments
bonus
deferred_income
total_stock_value
expenses
exercised_stock_options
other
long_term_incentive
restricted_stock
```

Email features
```
to_messages
from_poi_to_this_person
from_messages
from_this_person_to_poi
shared_receipt_with_poi
```

### Performance Review: remove features with more than 60% NaNs

Here are the new metrics obtained by removing the features with more than 60% missing values, that were not adding value and could potentially cause noise in the results.

AdaBoost
```
Accuracy: 0.83279
Precision: 0.38580
Recall: 0.28800
```

RandomForest
```
Accuracy: 0.84536
Precision: 0.36967
Recall: 0.11700
```

SVC
```
Accuracy: 0.50900
Precision: 0.15763
Recall: 0.56100
```

### Feature Engineering
New financial features
```
f_bonus = bonus / total_payments
f_salary = salary / total_payments
f_stock = total_stock_value / total_payments
```
The first set of engineered features relates to the fraction (f stands for fraction) from the type of financial incentives received. 
Employees usually can be rewarded mainly through three mechanisms: salary, bonus or stock.

New email features
```
r_from = from_this_person_to_poi / from_messages
r_to = from_poi_to_this_person / to_messages
```
The second set of engineered features relates to the ratio (r stands for ratio) of email sent to or received from a POI. Since the total data for to and from is available, getting the ratio is rather easy.

### Performance Review: adding engineered features
AdaBoost
```
Accuracy: 0.83729
Precision: 0.40479
Recall: 0.29550
```
RandomForest
```
Accuracy: 0.85271
Precision: 0.44404
Recall: 0.12300
```
SVC
```
Accuracy: 0.49636
Precision: 0.14772
Recall: 0.52950
```

### Tuning

### RandomForest and AdaBoost
The tune process for both algorithms has been similar, and the results almost identical.

They were tested using rf_tune() and ab_tune() respectively. Tests were performed with and without scaler using StandardScaler, it made little difference in RandomForest and no difference in AdaBoost.

### SVC
SVC went through a similar optimization process, but with more focus on the parameters, since Pipeline allowed more customization when it came to this algorithm.

Also high expectations were held for how the pre-processing through feature scaling would affected its final results, since it had little to no effect to AdaBoost and RandomForest.

As expected, the scaler and parameter tuning in SVC had a huge positive impact.

```
Accuracy: from 0.39936 to 0.80336
Precision: from 0.12769 to 0.32545
Recall: from 0.54950 to 0.35100
```

Incredibly, both Accuracy and Precision more than doubled its score, on the other hand Recall got more in line with Precision. 

### Conclusion

Interestingly enough, SVC, the worst performer along all the project, turned the situation in its head by using a scaler and fine tuning its parameters.

As the table above shows, SVC has the most healthy balance across Precision and Recall, with a Accuracy almost at par with AdaBoost and RandomForest.




