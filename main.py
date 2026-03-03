import os 
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import cross_val_score,GridSearchCV

MODEL_FILE="model.pkl"

def build_pipline(num_attribs,cat_attribs):
    num_pipeline=Pipeline(
        [
            ('imputer',SimpleImputer(strategy='median')),
            ('scaler',StandardScaler())
        ]
    )

    cat_pipeline=Pipeline([
        ('encoding',OneHotEncoder(handle_unknown='ignore'))
    ])

    full_pipeline=ColumnTransformer([
        ('num',num_pipeline,num_attribs),
        ('cat',cat_pipeline,cat_attribs)
    ])

    return full_pipeline

if not os.path.exists(MODEL_FILE):
    housing=pd.read_csv('housing.csv')
    housing['income_cat']=pd.cut(housing['median_income'],bins=[0.0,1.5,3.0,4.5,6.0,np.inf],
                                 labels=[1,2,3,4,5])
    split=StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)
    
    for train_index,test_index in split.split(housing,housing['income_cat']):
        housing.loc[test_index].drop('income_cat',axis=1).to_csv('input.csv',index=False)
        housing=housing.loc[train_index].drop('income_cat',axis=1)
     
    housing_label=housing['median_house_value'].copy()
    housing_feature=housing.drop("median_house_value",axis=1)

    num_attribs=housing_feature.drop('ocean_proximity',axis=1).columns.to_list()
    cat_attribs=['ocean_proximity']

    pipeline=build_pipline(num_attribs,cat_attribs)
    full_pipeline = Pipeline([
        ("preparation", pipeline),
        ("model", RandomForestRegressor(random_state=42))
    ])

    # Hyperparameter Grid
    param_grid = {
        "model__n_estimators": [100],
        "model__max_depth": [10,None],
        "model__max_features": [4, 6]
    }

    grid_search = GridSearchCV(
        full_pipeline,
        param_grid,
        cv=5,
        scoring="neg_mean_squared_error",
        n_jobs=-1
    )

    grid_search.fit(housing_feature, housing_label)

    best_model = grid_search.best_estimator_

    print("Best Parameters Found:")
    print(grid_search.best_params_)


    # save model and pipline
    joblib.dump(best_model,MODEL_FILE)

else:
    model=joblib.load(MODEL_FILE)
    # pipeline=joblib.load(PIPELINE)

    input_data=pd.read_csv('input.csv')
    # tranform_data=pipeline.transform(input_data)
    prediction=model.predict(input_data)
    input_data['median_house_value']=prediction
    input_data.to_csv('output.csv',index=False)
    print("Interfernce is complete result are save in output.csv ")