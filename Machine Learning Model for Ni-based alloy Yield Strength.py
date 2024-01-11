import numpy as np
# Import data file
import matplotlib.pyplot as plt
import pandas as pd
df = pd.read_excel(r"C:\Users\jcao1\Desktop\Machine Learning\0_data_cast_collection_original.xlsx") 

# Keeping reference will influence the judgement of output
del df['reference'] 

# The targets can be the relatinship between Yield Strength and Elongation percentage.
del df['UTS/Mpa']

# Drop any value that targets is N/A
df.dropna(how='any', inplace=True)

# Classify features and targets and seperate them
feature_labels = "test_T,Ni (wt.%),Co (wt.%),Cr (wt.%),Mo (wt.%),Ti (wt.%),Al (wt.%),C (wt.%),Nb (wt.%),Ta (wt.%),Zr (wt.%),W (wt.%),B (wt.%),V (wt.%),Fe (wt.%),Si (wt.%),Mn (wt.%),Hf (wt.%),solution_step_1_T/℃,solution_step_1_t/h,solution_step_2_T/℃,solution_step_2_t/h,cooling_rate/℃/s,aging_step1_T/℃,aging_step1_t/h,aging_step2_T/℃,aging_step2_t/h".split(",")
target_labels = "YS/Mpa,EL/%".split(",")
features, targets = df[feature_labels], df[target_labels]
d = 'solution_step_1_t/h,solution_step_2_T/℃,solution_step_2_t/h,aging_step1_T/℃,aging_step1_t/h,aging_step2_T/℃'.split(",")
noise = df[d] 

# Split the dataset into training set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( features, targets, test_size=0.33, random_state=42) 

# Use Recursive feature elimination to select the model, using Wrappers methods, using a for loop to test the relationsip between number of features selected and model accuracy, also standardlize X_feature using MinMax scaler
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
scaler = MinMaxScaler()
X = scaler.fit(X_train)
scaler.set_output(transform="pandas")
X_train = scaler.fit_transform(X_train)
empty = [] # Append r2 score for each target's accuracy
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state=0)

for i in range(0,27):
   

    
    estimator = SVR(kernel="linear")
    selector = RFE(estimator, n_features_to_select= 27 - i, step=1)
# Set it to 1, the recursive feature elimination is going to remove one feature through each iteration, step corresponds to the (integer) number of features to remove at each iteration.
    selector.fit(X_train, y_train['YS/Mpa'].values.reshape(-1, 1))
    selected_features1 = X_train.columns[(selector.get_support())]
    x = list(selected_features1.values)
    common = list(set(x) & set(noise))
    for i in common:
        x.remove(i)
    features = df[x]
    features = pd.concat([features, noise.reindex(features.index)], axis=1)
# Set up a training model to test the evaluation, using SVR which is used to find a vector platform where has the shortest distance between all the features in a set
    X_train, X_test, y_train, y_test = train_test_split( features, targets, test_size=0.33, random_state=42)
    scaler.set_output(transform="pandas")
    X_train = scaler.fit_transform(X_train)
    
    regressor.fit(X_train,y_train['YS/Mpa'].values.reshape(-1, 1))
    X_test = scaler.fit_transform(X_test)
    y_pred1 = regressor.predict(X_test)
    
    from sklearn.metrics import r2_score  # Evaluate the number of  using r2 score
    r2_a = r2_score(y_pred1, y_test['YS/Mpa'])
    empty.append(r2_a)
empty = empty[::-1]
print(empty)
a = np.array(empty[1:21])
b = np.arange(7,27)


fig, ax = plt.subplots()
plt.plot(b,a)
plt.xlabel('Number of features')
plt.ylabel('R2 Score')
plt.title('R2 Score Evaluation for YS Feature Selection')
plt.yticks(fontproperties = 'Times New Roman', size = 20)
plt.xticks(fontproperties = 'Times New Roman', size = 20)
def annot_max(x,y, ax=None):
    xmax = x[np.argmax(y)]
    ymax = y.max()
    text= "x={:.3f}, y={:.3f}".format(xmax, ymax)
    if not ax:
        ax=plt.gca()
    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
    arrowprops=dict(arrowstyle="->",connectionstyle="angle,angleA=0,angleB=60")
    kw = dict(xycoords='data',textcoords="axes fraction",
              arrowprops=arrowprops, bbox=bbox_props, ha="right", va="top")
    ax.annotate(text, xy=(xmax, ymax), xytext=(0.94,0.96), **kw)

annot_max(b,a)
plt.show()
print(max(empty), (empty.index(max(empty))*1)+6)  # Find the largest r2 score and how many elements are in there

best_selected_features = 'test_T, Ni (wt.%), Co (wt.%), Cr (wt.%), Al (wt.%), Ta (wt.%) ,Zr (wt.%), W (wt.%), B (wt.%), Si (wt.%), Mn (wt.%), Hf (wt.%) ,solution_step_1_T/℃, cooling_rate/℃/s, aging_step2_t/h, solution_step_1_t/h, solution_step_2_T/℃,solution_step_2_t/h ,aging_step1_T/℃, aging_step1_t/h,aging_step2_T/℃'.split(",")

# Ni has been mannually fitlered because it acts as a base-elements in the alloy

best_selected_features = "test_T,Co (wt.%),Cr (wt.%),Al (wt.%),Ta (wt.%),Zr (wt.%),W (wt.%),B (wt.%),Si (wt.%),Mn (wt.%),Hf (wt.%),solution_step_1_T/℃,cooling_rate/℃/s,aging_step2_t/h,solution_step_1_t/h,solution_step_2_T/℃,solution_step_2_t/h,aging_step1_T/℃,aging_step1_t/h,aging_step2_T/℃".split(",")
# Hence the number of features selected is 20, which has a coreesponding r2 score of 0.8497640902102799
best_selected_features = df[best_selected_features]



# Evaluate the model using cross validation set
X_train, X_test, y_train, y_test = train_test_split( best_selected_features , targets, test_size=0.33, random_state=41) 

from sklearn.model_selection import cross_val_score, KFold
# The model selection candidates are import as follow 
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.kernel_ridge import KernelRidge

# Apply Grid Search to each model, since linear regression doesn't have any hyperparameters, hence it is not needed for grid search
'''
# Random Forest Regressor
from sklearn.model_selection import RandomizedSearchCV
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf = RandomForestRegressor()
# Random search of parameters, using 5 fold cross validation, 
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 5, verbose=2, random_state=42, n_jobs = -1)
# Fit the random search model
rf_random.fit(X_train, y_train['YS/Mpa'])
a = rf_random.best_params_ # Selected hyper parameters
# The selected hyper parameters are following: {'n_estimators': 1400, 'min_samples_split': 2, 'min_samples_leaf': 1, 'max_features': 'auto', 'max_depth': 100, 'bootstrap': True}


# Gradient Boosting Regressor
boost_grid = {
    'n_estimators': np.arange(50, 251, 50),          # Number of boosting stages to be run
    'learning_rate': [0.01, 0.1, 0.2, 0.3],         # Step size shrinkage to prevent overfitting
    'max_depth': np.arange(3, 11),                  # Maximum depth of the individual trees
    'min_samples_split': np.arange(2, 11),          # Minimum number of samples required to split an internal node
    'min_samples_leaf': np.arange(1, 11),           # Minimum number of samples required to be at a leaf node
}
gb = GradientBoostingRegressor(random_state=42)
random_search = RandomizedSearchCV(estimator = gb, param_distributions  = boost_grid, n_iter=100, cv=5, verbose=2, random_state=42,n_jobs = -1)
random_search.fit(X_train, y_train['YS/Mpa'])
b = random_search.best_params_ # Selected hyper parameters 
# The selected hyper parameters are following: {'n_estimators': 100, 'min_samples_split': 7, 'min_samples_leaf': 1, 'max_depth': 9, 'learning_rate': 0.3}


# Decision Tree Regressor
tree_grid = {
    'max_depth': np.arange(1, 11),                 # Maximum depth of the tree
    'min_samples_split': np.arange(2, 11),         # Minimum number of samples required to split an internal node
    'min_samples_leaf': np.arange(1, 11),          # Minimum number of samples required to be at a leaf node
    'max_features': ['auto', 'sqrt', 'log2'],      # Number of features to consider when looking for the best split
}

tree = DecisionTreeRegressor(random_state=42)
tree_search = RandomizedSearchCV(estimator = tree, param_distributions = tree_grid, n_iter=100, cv=5,verbose = 2, random_state=42,n_jobs = -1)
tree_search.fit(X_train, y_train['YS/Mpa'])
c = tree_search.best_params_ # Selected hyper parameters 
# The selected hyper parameters are following: {'min_samples_split': 4, 'min_samples_leaf': 2, 'max_features': 'auto', 'max_depth': 9}

# Kernel Ridge Regressor
ridge_grid = {
    'alpha': np.logspace(-3, 3, 7),      # Regularization parameter
    'kernel': ['linear', 'poly', 'rbf'], # Kernel types
    'gamma': ['scale', 'auto'] + list(np.logspace(-3, 3, 7))  # Kernel coefficient
}
# Create the KernelRidge model
krr = KernelRidge()

# Initialize RandomizedSearchCV
krr_search = RandomizedSearchCV(estimator=krr, param_distributions=ridge_grid, n_iter=100, cv=5, verbose = 2, random_state=42,n_jobs = -1)

# Perform random search on the data
krr_search.fit(X_train, y_train['YS/Mpa'])
d = krr_search.best_params_ # Selected hyper parameters 
# The selected hyper parameters are following: {'kernel': 'linear', 'gamma': 1000.0, 'alpha': 10.0}
'''

from sklearn.metrics import mean_squared_error   # r2 score and Mean Squared Error has been used to score the model    
from sklearn.base import clone
# Initialize the models
model_linear = LinearRegression()
model_linear = clone(model_linear)
model_rf = RandomForestRegressor(n_estimators=1400,max_depth=100,min_samples_split=2,min_samples_leaf=1,max_features='auto',bootstrap=True)
model_rf = clone(model_rf)
model_gb = GradientBoostingRegressor(learning_rate=0.3, n_estimators=100,min_samples_split=7, min_samples_leaf=1,max_depth=3)
model_gb = clone(model_gb)
model_tree = DecisionTreeRegressor(max_depth=9, min_samples_split=4, min_samples_leaf=2,max_features='auto')
model_tree = clone(model_tree)
model_krr = KernelRidge(alpha=10.0,kernel='linear', gamma=1000.0)
model_krr = clone(model_krr)


model = [model_linear,model_rf,model_gb,model_tree,model_krr]

# Initialize scoring board
r2_scores_train = []
mean_squared_error_train = []


# Perform cross-validation using r2 score
num_folds = 5
kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
for i in model:
    value = cross_val_score(i,X_train,y_train['YS/Mpa'], scoring = 'r2', cv=kf) 
    value = value.mean()
    value = round(value,3)
    r2_scores_train.append(value)
    


# Perform cross-validation using negative mean squared error score
for i in model:
    value = cross_val_score(i, X_train,y_train['YS/Mpa'], scoring = 'neg_mean_squared_error', cv=kf) 
    value = value.mean()
    mean_squared_error_train.append(abs(value))

# Train each model and evaluate each of them using r2 score
imp = []
for i in model:
    i.fit(X_train,y_train['YS/Mpa'])
    y_pred = i.predict(X_test)
    r2_value = mean_squared_error(y_pred,y_test['YS/Mpa'])
    imp.append(r2_value)

model = ['Linear Regression','Random Forest Regressor','Gradient Boosting Regressor','Decision Tree Regressor','Kernel Ridge']
'''
import matplotlib.pyplot as plt
x = np.arange(len(model))
width = 0.35
fig,ax = plt.subplots()
bar1 = ax.bar(x- width * 0.5,  mean_squared_error_train, width, label = 'R2 scores for training set')
bar2 = ax.bar(x + width * 0.5,  imp, width, label = 'R2 scores for test set')
ax.set_xticks(x)
ax.set_xticklabels(model)
ax.legend()
ax.bar_label(bar1, padding=3)
ax.bar_label(bar2, padding=3)
plt.xlabel('Model')
plt.ylabel('Mean Squared Error')
plt.title('Mean Squared Error evaluation for selected models')
plt.show()
'''
# By r2 score and mean squared error, gradient boost regressor has been selected
# Apply grid search for Gradient Boost regressor based on its random search result
from sklearn.model_selection import GridSearchCV
# Create the parameter grid based on the results of random search 

# The selected hyper parameters are following: {'n_estimators': 100, 'min_samples_split': 7, 'min_samples_leaf': 1, 'max_depth': 9, 'learning_rate': 0.3}
boost_grid = {
     'n_estimators': [80, 100, 120],
    'min_samples_split': [5, 7, 9],
    'min_samples_leaf': [1, 2],
    'max_depth': [8, 9, 10],
    'learning_rate': [0.2, 0.3, 0.4]
}
gbr = GradientBoostingRegressor()
grid_search = GridSearchCV(gbr, param_grid = boost_grid, cv=5, n_jobs=-1)
grid_search.fit(X_train, y_train['YS/Mpa'])
parameters = grid_search.best_params_
# Parameters are same as random search selected parameters {'n_estimators': 100, 'min_samples_split': 7, 'min_samples_leaf': 1, 'max_depth': 9, 'learning_rate': 0.3}


validation = best_selected_features.loc[0:4]
best_selected_features.drop(best_selected_features.index[[0,1,2,3,4]], inplace=True)
targets.drop(best_selected_features.index[[0,1,2,3,4]], inplace=True)
# Train the model
X_train, X_test, y_train, y_test = train_test_split( best_selected_features, targets, test_size=0.33, random_state=42) 
gbr = GradientBoostingRegressor(learning_rate=0.1,n_estimators=59,min_samples_split=11, min_samples_leaf=1,max_depth=5)
gbr.fit(X_train,y_train['YS/Mpa'])
y_predict_train = gbr.predict(X_train)
y_predict_test = gbr.predict(X_test)
print(y_train['YS/Mpa'].values)
print(y_test['YS/Mpa'].values)
a = r2_score(y_predict_train,y_train['YS/Mpa'].values)
print(a)
b =  r2_score(y_predict_test,y_test['YS/Mpa'].values)
print(b)
def error_term(x,y):
    value = (y-x)/ y
    return abs(value)


validation_value = targets.loc[0:4]
validation_value = validation_value['YS/Mpa'].values
print(validation_value)

y_validation = gbr.predict(validation)
s = []
for i in range(4):
    g = error_term(y_validation[i],validation_value[i])
    s.append(g)
print(s)
