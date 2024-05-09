import shap
import pandas as pd
import numpy as np
from sklearn .model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from  sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import neighbors
from sklearn import svm
from BorutaShap import BorutaShap

class ModelProcess:
    T1_id = {
            'Mental' : [
                'T1_PONT_TOTAL',
                'T1_PONT_A',
                'T1_PONT_D'
            ], 
            'Avaliacao':[
                'PESO_T1','IMC_T1','IMC1_CLASS','CC_T1',
                'CC_T1_CLAS','CQ_T1','RCQ_T1','RCQ_T1_CLAS',
                'RCE_T1','RCE_T1_CLAS','CP_T1','CP_T1_CLAS',
                'CB_T1','CB_T1_ADEQUA','CB_T1_CLAS'
            ], 
            'Dietetico':[
                'VCT_T1','CHO_G_T1','CHO_%_T1','PTN_G_T1',
                'PTN_%_T1','LIP_G_T1','LIP_%_T1','COLESTEROL_T1',
                'ADEQ_COLES_T1','G_SAT_T1','ADEQ_G_SAT_T1'
            ],
            'Questionario':[
                'Q_1_T1','Q_2_T1','Q_3_T1','Q_4_T1','Q_5_T1',
                'Q_6_T1','Q_7_T1','Q_8_T1','Q_9_T1','Q_10_T1',
                'Q_11_T1','Q_12_T1','Q_13_T1','Q_14_T1','Q_15_T1',
                'Q_16_T1','Q_17_T1','Q_18_T1','Q_19_T1','Q_20_T1',
                'Q_21_T1','Q_22_T1','Q_23_T1','TOTAl_T1'
            ]
        }
    T2_id = {
        'Mental' : [
            'T2_PONT_TOTAL',
            'T2_PONT_A','T2_PONT_D'
        ],
        'Avaliacao':[
            'PESO_T2','IMC_T2','IMC2_CLASS','CC_T2',
            'CC_T2_CLAS','CQ_T2','RCQ_T2','RCQ_T2_CLAS',
            'RCE_T2','RCE_T2_CLAS','CP_T2','CP_T2_CLAS',
            'CB_T2','CB_T2_ADEQUA','CB_T2_CLAS'

        ],
        'Dietetico':[
            'VCT_T2','CHO_G_T2','CHO_%_T2','PTN_G_T2',
            'PTN_%_T2','LIP_G_T2','LIP_%_T2','COLESTEROL_T2',
            'ADEQ_COLES_T2','G_SAT_T2','ADEQ_G_SAT_T2'
        ],
        'Questionario':[
            'Q_1_T2','Q_2_T2','Q_3_T2','Q_4_T2','Q_5_T2',
            'Q_6_T2','Q_7_T2','Q_8_T2','Q_9_T2','Q_10_T2',
            'Q_11_T2','Q_12_T2','Q_13_T2','Q_14_T2','Q_15_T2',
            'Q_16_T2','Q_17_T2','Q_18_T2','Q_19_T2','Q_20_T2',
            'Q_21_T2','Q_22_T2','Q_23_T2','TOTAl_T2'
        ]
    }
    Target ={
            "Classificatorias" :[
                'IMC2_CLASS',
                'CC_T2_CLAS',
                'RCQ_T2_CLAS',
                'RCE_T2_CLAS',
                'CP_T2_CLAS',
                'CB_T2_CLAS'
            ],
            'Numericas':[
                'IMC_T2',
                'CC_T2',
                'CQ_T2',
                'RCQ_T2',
                'RCE_T2',
                'CP_T2',
                'CB_T2',
                'CB_T2_ADEQUA'
            ]
        }
    def __init__(self,path : str = "data/staged/full_treated_data.json") -> None:
        self.df =  pd.read_json(path)
        

    def drop_empty(self,df: pd.DataFrame)->pd.DataFrame:
        return df[(df != 999).all(axis=1)].reset_index(drop=True)
    
    @property
    def data_type_class(self):
        """ Get data for identifying if pacient is from control or SOP Group"""
        cols = [
            'ID_VOLUNT', 
            ]
        cols.extend(self.T1_id['Mental'])
        cols.extend(self.T1_id['Questionario'])
        df = self.df[cols]
        df = self.drop_empty(df)
        return {
            "Y" : df['ID_VOLUNT'],
            "X" : df.drop(columns='ID_VOLUNT')
        }
    
    def instantiate_models(self):
        self.models ={
            'lr_clf' : LogisticRegression(random_state=0),
            'rf_clf' : RandomForestClassifier(random_state=0),
            'gb_clf' : GaussianNB(),
            'knn_clf' : neighbors.KNeighborsClassifier(),
            'svm_clf' : svm.SVC(kernel='linear'),
            'gbc_clf' : GradientBoostingClassifier(random_state=0),
            'dt_clf' : DecisionTreeClassifier(random_state=0)
        }
    
    def select_best_model(self):
        data = self.data_type_class
        X_train, X_test, y_train, y_test = train_test_split(data['X'], data["Y"], random_state=0)
        best_model = { 
            "Name" : None,
            "Score" : 0.0,
            "Model" : None
            }
        for name, model in self.models.items():
            clf = model.fit(X_train,y_train)
            score = clf.score(X_test, y_test)
            # print(clf.feature_importances_)
            if score > best_model["Score"]:
                best_model = { 
                    "Name" : name,
                    "Score" : score,
                    "Model" : clf
                    }
        return best_model
    
    def get_feature_importance(self, rf_clf  : RandomForestClassifier) -> dict:
        features = rf_clf.feature_names_in_
        importances =  rf_clf.feature_importances_
        feature_importances = {fea: imp for imp, fea in zip(importances, features)}
        return  {k: v for k, v in sorted(feature_importances.items(), key=lambda item: item[1], reverse = True)}
    
    def print_importance(self, importance:dict)-> None:
        for k, v in importance.items():
            print(f"{k} -> {v:.4f}")

    def select_features(self):
        data = self.data_type_class
        X_train, X_test, y_train, y_test = train_test_split(data['X'], data["Y"], random_state=0)
        clf = self.models['rf_clf'].fit(X_train,y_train)
        explainer = shap.Explainer(clf.predict, X_test)
        importance =  self.get_feature_importance(clf)
        shap_values = explainer(X_test)
        from boruta import BorutaPy


        

            
if __name__ == "__main__":
    Process = ModelProcess()
    Process.instantiate_models()
    Process.select_features()
    