Hyperparameter Tuning
---

The attached Python code offers a template for tuning hyperparameters for a Random Forest model.
The code offers a step-by-step example of how you move from using the default hyperparameters to
using the tuned hyperparameters.

The template can be updated to tune the hyperparameters of multiple models at the same time via 
GridSearchCV. 

See an example below of how the hyperparameters of multiple models can be tuned at the same time.

```python
model_params = {

    'random_forest': {'model': RandomForestClassifier(criterion='entropy', random_state=1),
                      'params': {'n_estimators': [5, 100, 200, 500, 1000], 'max_features': ['auto', 'log2'],
                                 'min_samples_leaf': [1, 2, 4], 'min_samples_split': [2, 5, 10]}},

    'knn': {'model': KNeighborsClassifier(algorithm='kd_tree'),
            'params': {'n_neighbors': [5, 10, 15, 25, 50, 100]}
            }}   
print(model_params)

scores = []
all_scores = []

# Fit the model_params to the GridSearch below.

for model_name, mp in model_params.items():
    clf = GridSearchCV(mp['model'], mp['params'], n_jobs=-1, scoring='f1_macro', cv=10,
                       return_train_score=True, verbose=2)
    clf.fit(X_train, y_train)
    scores.append({
        'model': model_name,
        'best_score': clf.best_score_,
        'best_params': clf.best_params_
    })
    all_scores.append({
        'model': model_name,
        'avg_score': clf.cv_results_['mean_test_score'],
        'std_test_score': clf.cv_results_['std_test_score'],
        'params': clf.cv_results_['params']
    })

print(scores)  
print(all_scores)
```
