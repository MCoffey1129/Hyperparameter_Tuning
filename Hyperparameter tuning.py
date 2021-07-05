
"""Below sets out a template I use for tuning hyperparameters. The code can be updated to tune the
   hyperparameters for multiple models at the same time via a GridSearch."""


"""Import packages"""
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report


"""# Import the iris dataset"""
iris = sns.load_dataset("iris")

"""Typical queries used to evaluate your data - always carry this out before completing any analysis
    on your data"""
iris.head()
iris.info()
iris.describe()
iris.columns
iris.isnull().sum() # there are no null values in the data

"""Check the correlation between each of the vars"""
"""Sepal length and Sepal width as well as petal length and petal with look to be highly correlated"""
sns.pairplot(iris)
iris.corr()

################################################################################################################
                # Random Forest used to predict the species
################################################################################################################

"""# Importing the dataset"""
X = iris.iloc[:, :-1].values
y = iris.iloc[:, -1].values

"""# Splitting the dataset into the Training set and Test set"""
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 1)
print(X_train)
print(y_train)
print(X_test)
print(y_test)

"""# Feature Scaling"""
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
print(X_train)
print(X_test)

"""# Training the Random Forest model on the Training data (without tuning the hyperparameters)"""
classifier = RandomForestClassifier(random_state = 1)
classifier.fit(X_train, y_train)

"""# Predicting the Test set results"""
y_pred = classifier.predict(X_test)

"""# Making the Confusion Matrix"""
"""Overall accuracy is 97% (please note F1 score is usually a better indicator of the success of the model
   especially if we have unbalanced classes)
   F1 score is also 97% 
   There was only one case which was misclassified"""
cm = confusion_matrix(y_test, y_pred)
print(cm)
cr = classification_report(y_test, y_pred)
print(cr)
accuracy_score(y_test, y_pred)


################################################################################################################
                # Tuning the hyperparameters - as our predicitions were 97% accurate we will not
                # gain anything from tuning the hyperparameters in this example
################################################################################################################



"""Model Parameters run through the GridSearch CV
   for more information on what these hyperparameters mean please visit the sklearn website 
   https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html"""


model_params = {

    'random_forest': {'model': RandomForestClassifier(criterion='entropy', random_state=1),
                      'params': {'n_estimators': [5, 100, 200, 500, 1000], 'max_features': ['auto', 'log2'],
                                 'min_samples_leaf': [1, 2, 4], 'min_samples_split': [2, 5, 10]}}

}

print(model_params)



"""Fit the model_params to the GridSearch below.
   We will set n_jobs = -1 to ensure that all the processors are used in the GridSearch
   Scoring is set to F1 instead of accuracy (although it does not matter in this instance)
   We complete 10 cross validations in order to ensure that the parameters perform well on unseen data"""

scores = []
for model_name, mp in model_params.items():
    clf = GridSearchCV(mp['model'], mp['params'], n_jobs=-1, scoring='f1_macro', cv=10,
                       return_train_score=True, verbose=2)
    clf.fit(X_train, y_train)
    scores.append({
        'model': model_name,
        'best_score': clf.best_score_,
        'best_params': clf.best_params_
    })

print(scores)
# [{'model': 'random_forest', 'best_score': 0.9631746031746031,
# 'best_params': {'max_features': 'auto', 'min_samples_leaf': 1, 'min_samples_split': 10, 'n_estimators': 200}}]


"""Let's fit the above best hyperparameters to the test data to see how it performs"""
classifier_tuned = RandomForestClassifier(criterion='entropy', max_features = 'auto', min_samples_leaf = 1,
                                          min_samples_split = 10, n_estimators = 200, random_state = 1)
classifier_tuned.fit(X_train, y_train)

"""# Predicting the Test set results"""
y_pred_tuned = classifier_tuned.predict(X_test)

"""# Making the Confusion Matrix"""
"""Overall accuracy is 97% (please note F1 score is usually a better indicator of the success of the model
   especially if we have unbalanced classes)
   F1 score is also 97% 
   There was only one case which was misclassified
   No increase in F1 or accuracy given that we only had one misclassified case previously"""
cm = confusion_matrix(y_test, y_pred_tuned)
print(cm)
cr = classification_report(y_test, y_pred_tuned)
print(cr)
accuracy_score(y_test, y_pred_tuned)

