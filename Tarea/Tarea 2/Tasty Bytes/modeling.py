import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

class Model():
    def __init__(self, mdls, target_names):
        self.mdls = mdls
        self.ds = None
        self.models = None
        self.le = None
        self.results = None
        self.target_names = target_names
        
    def preprocess(self, df, target):
        # Get the log transformed dataset and separate features from target
        X = df.drop(target, axis=1)
        y = df[targtet]
        
        np.random.seed(42)
        # split between train and test
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2, stratify=y)
        
        # encode the target variable
        le = LabelEncoder()
        y_train_encoded = le.fit_transform(y_train)
        y_test_encoded = le.transform(y_test)
        y_encoded = le.transform(y)
        
        
        np.random.seed(42)

        # selector for categorical and numerical features
        categorical_features = make_column_selector(dtype_include='object')
        numeric_features = make_column_selector(dtype_include=np.number)

        # pipeline numerics (we do not need the simple imputer but could be good to have it for impute new data)
        numeric_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")), 
            ("scaler", StandardScaler())
        ])

        # pipeline for categorical, one hot encode and select features
        categorical_transformer = Pipeline(steps=[
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
            ("selector", SelectPercentile(chi2, percentile=35))
        ])

        # now union the column transformers
        preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ])

        ds = [
           ('global', X, y_encoded), 
           ('train', X_train, y_train_encoded), 
           ('test', X_test, y_test_encoded)
        ]

        self.ds = ds
        self.preprocessor = preprocessor
        self.le = le
    
        
    def fit(self):
        mdls = self.mdls
        # perform training/fitting
        self.models = dict()
        for model_name, model in mdls:
            steps = [("preprocessor", preprocessor), ("model", model)]    
            pipe = Pipeline(steps=steps)
            print(f'Fitting model {model_name}')
            pipe.fit(X_train, y_train_encoded)
            self.models[model_name] = pipe
            
    def eval(self, threshold=0.5):
        mdls = self.mdls
        ds = self.ds
        target_names = self.target_names
        
        results = {model_name:{dsname[0]:{} for dsname in ds} for model_name, _ in mdls}
        for model_name, model in models.items():
            for dsname, features, gt in ds:
                print(f'Predicting on model {model_name} over {dsname} dataset')
                y_pred = model.predict_proba(features)[:, 1] > threshold
                results[model_name][dsname] = {'pred':y_pred}
                results[model_name][dsname] = {'metrics': {model_name:{'precision': precision_score(gt, y_pred),
                                                                       'recall': recall_score(gt, y_pred),
                                                                       'f1': f1_score(gt, y_pred)
                                                                      }
                                                          },
                                               'cm': confusion_matrix(gt, y_pred),
                                               'rep': pd.DataFrame(classification_report(gt, y_pred, 
                                                                                         target_names=target_names,
                                                                                         output_dict=True))
                                              }
        self.results = results