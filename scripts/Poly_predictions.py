import pandas as pd
import seaborn as sns
import tabulate
import warnings
import logging
from sklearn .model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from  sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import neighbors
from sklearn import svm
from sklearn.linear_model import RidgeCV
import matplotlib.pyplot as plt
import numpy as np

class Runner():
    def __init__(self, path : str = "data/raw/survey.xlsx") -> None:
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        self.logger: logging.Logger = logging.getLogger(__name__)
        try:
            self.df_raw = pd.read_excel(path, header = 1)
            self.logger.info('Raw data secured.')
        except FileNotFoundError:
            raise( "No survey file in folder")
        self.col_id = ['ID_PROJ','IDADE','ALTURA']
        self.smart_id = ['SEGUIU_METAS_1','SEGUIU_METAS2','SEGUIU_METAS3']
        self.T1_id = {
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
        self.T2_id = {
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
        self.Target ={
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
        self.fix_smart()
        self.fix_mental()
        self.fix_Q()
        self.fix_diet()
        self.fix_peso()
        self.fix_cb_class()
        self.fix_imc_class()
        self.fix_avaliação()

    def drop_empty(self,df: pd.DataFrame)->pd.DataFrame:
        # df = df.dropna()
        return df[(df != 999).all(axis=1)].reset_index(drop=True)
    
    def select_features(self,N : int,cleaned_data) -> list:
        data = cleaned_data
        X = data['X']
        y = data['Y']
        ridge = RidgeCV(alphas=np.logspace(-6, 6, num=5)).fit(X, y)

        importance = np.abs(ridge.coef_)
        feature_names = np.array(X.columns)

        # Sort the features by their importance
        indices = np.argsort(importance)[::-1]
        top_feature_indices = indices[:]
        top_features = feature_names[top_feature_indices]
        top_importance = importance[top_feature_indices]
        plt.clf()
        plt.barh(top_features, top_importance)
        plt.xlabel('Importance')
        plt.title("Feature Importances via Coefficients")
        plt.gca().invert_yaxis()  # Invert y-axis to have the most important feature on top
        plt.tight_layout()  # Adjust layout to prevent clipping of labels
        plt.savefig(f'images/Type_1/Data_Behavior/Feature_Importance.png')
        return top_features[:N].tolist()
    
    def data_per_target(self):
        df = self.df_raw#[self.df_raw["ID_VOLUNT"] != 1]
        print(df.columns.tolist())
        df = df.drop(columns=[
            # 'ID_PROJ',
            'INICIAIS'])
        print(df.shape)
        keys_t1 = [
            'Mental',
            'Avaliacao',
            'Dietetico',
            'Questionario'
            ]
        keys_t2 = [
            'Mental',
            'Avaliacao',
            'Dietetico',
            'Questionario'
            ]
        cols = ['ID_PROJ', 'ID_VOLUNT', "ALTURA",'IDADE ']
        for sublist in self.smart_id:
            cols.append(sublist)
        for key in keys_t1:
            for sublist in self.T1_id[key]:
                cols.append(sublist)
        for key in keys_t2:
            for sublist in self.T2_id[key]:
                 cols.append(sublist)
        df : pd.DataFrame = df[cols]
        # print(df)
        # df = self.drop_empty(df)

        df.to_json("data/staged/full_treated_data.json")
    def fix_peso(self):
        mask = self.df_raw["PESO_T1"] != 999
        self.df_raw.loc[mask & (self.df_raw["PESO_T2"] == 999), "PESO_T2"] = self.df_raw.loc[mask, "PESO_T1"]

    def classify_cb(self,value):
        if value <34.0:
            return 1
        elif value < 37.5:
            return 2
        elif value > 37.5 and value < 999.0:
            return 3
        else:
            return 999.0
        
    def fix_cb_class(self):
        self.df_raw['CB_T1_CLAS'] = self.df_raw['CB_T1'].apply(lambda x: self.classify_cb(x))
        self.df_raw['IMC_T2'] = self.df_raw['CB_T2'].apply(lambda x: self.classify_cb(x))
    
    def classify_imc(self,value):
        if value <25:
            return 5
        elif value < 30:
            return 4
        elif value <  35 :
            return 3
        elif value <  40 :
            return 2
        elif value > 40 and value < 999.0:
            return 1
        else:
            return 999
        
    def fix_imc_class(self):
        self.df_raw['IMC1_CLASS'] = self.df_raw['IMC_T1'].apply(lambda x: self.classify_imc(x))
        self.df_raw['IMC2_CLASS'] = self.df_raw['IMC_T2'].apply(lambda x: self.classify_imc(x))

    def fix_avaliação(self):
         for key_1, key_2 in zip(self.T1_id['Avaliacao'], self.T2_id['Avaliacao']):
            mask = self.df_raw[key_1] != 999
            self.df_raw.loc[mask & (self.df_raw[key_2] == 999), key_2] = self.df_raw.loc[mask, key_1]

    def fix_smart(self)->None:
        for col in self.smart_id:
            self.df_raw[col] = self.df_raw[col].map({1.0 : 1.0, 
                                                     2.0 : 0.5, 
                                                     0.0 : 0.0,
                                                     999.0:999.0,
                                                     np.NaN : np.NaN 
                                                     })
    def fix_diet(self):
        for key_1, key_2 in zip(self.T1_id['Dietetico'], self.T2_id['Dietetico']):
            mask = self.df_raw[key_1] != 999
            self.df_raw.loc[mask & (self.df_raw[key_2] == 999), key_2] = self.df_raw.loc[mask, key_1]
    def fix_mental(self):
        for key_1, key_2 in zip(self.T1_id['Mental'], self.T2_id['Mental']):
            mask = self.df_raw[key_1] != 999
            self.df_raw.loc[mask & (self.df_raw[key_2] == 999), key_2] = self.df_raw.loc[mask, key_1]
    def fix_Q(self):
        for key_1, key_2 in zip(self.T1_id['Questionario'], self.T2_id['Questionario']):
            mask = self.df_raw[key_1] != 999
            self.df_raw.loc[mask & (self.df_raw[key_2] == 999), key_2] = self.df_raw.loc[mask, key_1]
        # for col in self.T1_id["Questionario"]:
        #         if "TOTAl_T1" != col:
        #             self.df_raw[col] =  self.df_raw[col].map({1.0 : 1.0, 2.0 : 2.0, 3.0 : 3.0, 4.0 : 4.0, 5.0:5.0, 999.0:0.0})
        #         else:
        #             self.df_raw[col] =  self.df_raw[col].apply(lambda x : int(str(x).replace('999','0')))
        # for col in self.T2_id["Questionario"]:
        #         if "TOTAl_T2" != col:
        #             self.df_raw[col] =  self.df_raw[col].map({1.0 : 1.0, 2.0 : 2.0, 3.0 : 3.0, 4.0 : 4.0, 5.0:5.0, 999.0:0.0})
        #         else:
        #             self.df_raw[col] =  self.df_raw[col].apply(lambda x : int(str(x).replace('999','0')))

    
            
if __name__ == '__main__':
    flux = Runner()
    flux.data_per_target()
