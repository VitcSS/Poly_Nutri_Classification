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

class Cleaner():
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
                # 'T1_PONT_A',
                # 'T1_PONT_D'
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
            'Status' : [
                'ID_VOLUNT'
            ],
            'Avaliacao':[
                'PESO_T2','IMC_T2','IMC2_CLASS','CC_T2',
                'CC_T2_CLAS','CQ_T2','RCQ_T2','RCQ_T2_CLAS',
                'RCE_T2','RCE_T2_CLAS','CP_T2','CP_T2_CLAS',
                'CB_T2','CB_T2_ADEQUA','CB_T2_CLAS'
            ]
        }
        self.setup()
    
    def drop_empty(self,df: pd.DataFrame)->pd.DataFrame:
        df = df.dropna()
        return df[(df != 999).all(axis=1)]
    
    def T1_data(self)-> dict:
        cols = []
        for sublist in (self.T1_id['Questionario'],self.T1_id['Mental']):
            cols = self.T1_id['Questionario']
            # cols.extend(sublist)
        df = pd.concat([self.df_raw[self.Target['Status']],self.df_raw[cols]], axis = 1)
        df.to_excel('text.xlsx')
        df = self.drop_empty(df)
        return {
            'X' : df[cols],
            'Y' : df[self.Target['Status'][0]]
        }
    
    def setup(self):
        self.cleaned_data = self.T1_data()

    def run_models(self, N = 10):
        cleaned_data = self.cleaned_data.copy()
        features = self.select_features(N)
        X_train,X_test,Y_train,Y_test = train_test_split(cleaned_data['X'][features],cleaned_data['Y'], test_size=0.3, random_state=0)
        print(cleaned_data['X'][features].shape)
        models ={
            'lr_clf' : LogisticRegression(random_state=0),
            'rf_clf' : RandomForestClassifier(random_state=0),
            'gb_clf' : GaussianNB(),
            'knn_clf' : neighbors.KNeighborsClassifier(),
            'svm_clf' : svm.SVC(kernel='linear'),
            'gbc_clf' : GradientBoostingClassifier(random_state=0),
            'dt_clf' : DecisionTreeClassifier(random_state=0)
        }
        best_model = {'Name': None, 'Score':0.0, 'Features': []}
        for model_name, model in models.items():
            trained = model.fit(X_train,Y_train)
            score = trained.score(X_test, Y_test)
            if score > best_model['Score']:
                best_model['Name'] = model_name
                best_model['Score'] = score
                best_model['Features'] = features
                best_model['Model'] = model
        return best_model

    def select_features(self,N : int) -> list:
        data = self.cleaned_data
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
        plt.savefig(f'images/Data_Behavior/Feature_Importance.png')
        return top_features[:N].tolist()
    
    def data_analysis_clf_status(self,features : list):
        df = pd.concat([self.cleaned_data['X'],self.cleaned_data['Y']], axis = 1)
        cols = features.copy()
        cols.append('ID_VOLUNT')
        df = df [cols].reset_index(drop=True)
        df_1 =  df[ df ['ID_VOLUNT'] == 1].drop(columns='ID_VOLUNT').reset_index(drop=True)
        df_2 =  df[ df ['ID_VOLUNT'] == 2].drop(columns='ID_VOLUNT').reset_index(drop=True)
        for column in cols:
            if 'Q_' in column:
                self.generate_histogram_Q(df_1,column,status = '1')
                self.generate_histogram_Q(df_2,column,status = '2')
            elif 'TOTAl_' in column:
                self.generate_histogram_cont(df_1,column,status = '1')
                self.generate_histogram_cont(df_2,column,status = '2')

    def generate_histogram_Q(self, df : pd.DataFrame, col, status):
        plt.clf()
        data : pd.DataFrame = df.copy()
        data[col] = data[col].astype('category')
        counts = df[col].value_counts().sort_index()
        # Plotting
        plt.bar(counts.index, counts.values)
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.title(f'Distribution of Categories for {col} where ID = {status}')
        plt.xticks(range(1, 6))
        plt.savefig(f'images/Data_Behavior/ID_{status}/{col}.png')
    
    def generate_histogram_cont(self, df : pd.DataFrame, col, status):
        plt.clf()
        data : pd.DataFrame = df[col].copy()
        iqr = np.percentile(data, 75) - np.percentile(data, 25)
        bin_width = 2 * iqr / (len(data) ** (1/3))
        num_bins = int((max(data) - min(data)) / bin_width)
        plt.hist(data, bins=20, edgecolor='black')  # Adjust the number of bins as needed
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.title(f'Distribution of Categories for {col} where ID = {status}')
        plt.savefig(f'images/Data_Behavior/ID_{status}/{col}.png')
        
    def confusion_matrix(self,best_model):
        model = best_model['Model']
        features = best_model['Features']
        name = best_model['Name']
        y_pred = model.predict(self.cleaned_data['X'][features])
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            confusion_matrix(self.cleaned_data['Y'], y_pred), 
            annot=True, 
            fmt='d', 
            cmap='Blues', 
            xticklabels=np.unique(self.cleaned_data['Y']), 
            yticklabels=np.unique(self.cleaned_data['Y'])
            )
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(f'Confusion Matrix for {name}')
        plt.savefig(f'images/Data_Behavior/Model_Behavior/Status_Classifier/Confusion_Matrix_{name}')


if __name__ == '__main__':
    flux = Cleaner()
    best_models = []
    for N in range(4,24):
        result = flux.run_models(N)
        best_models.append(result)
    best_model = max(best_models, key=lambda x: x.get('Score', 0))
    print(best_model)
    flux.confusion_matrix(best_model)
    flux.data_analysis_clf_status(best_model['Features'])
