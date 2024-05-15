
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn .model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.metrics import f1_score, roc_auc_score, log_loss, precision_score, recall_score
from  sklearn.linear_model import LogisticRegression
from sklearn.linear_model import RidgeCV
from sklearn.exceptions import ConvergenceWarning
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import neighbors
from sklearn import svm
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from string import Template
from collections import OrderedDict

class Process:
    def __init__(self,target : str, path : str = 'data/staged/full_treated_data.json', ) -> None:
        try:
            os.makedirs(f'results/{target}/')
        except:
            print("Folder already exists")
        self.df_raw = pd.read_json(path)
        self.target = target
        self.df_raw.columns = [str(x).strip() for x in self.df_raw.columns.to_list()]
        self.instantiate_models()
        self.df_raw['ID_VOLUNT'] = self.df_raw['ID_VOLUNT'].apply(lambda x : 0 if x == 2 else 1)
        self.fix_smart()
        self.fix_diet()
        self.fix_mental()
        print({
            'Controle Hígio': {
                'Original' : 1,
                'Atualizado' : 1
            },
            'SOP' : {
                'Original' : 2,
                'Atualizado' : 0
            }
        })
    
    @staticmethod
    def drop_empty(df: pd.DataFrame)->pd.DataFrame:
        return df[(df != 999).all(axis=1)].reset_index(drop=True).copy()
    
    @property
    def SOP_data(self) -> pd.DataFrame:
        return self.df_raw[self.df_raw['ID_VOLUNT']== 0].reset_index(drop=True).copy()
    
    @property
    def condition_quest_data(self) -> dict:
        target = 'ID_VOLUNT'
        cols = [
        # 'IDADE'
        ]
        for i in range(1,24):
            cols.append(f'Q_{i}_T1')
        cols.append("TOTAl_T1")
        all_cols = cols
        all_cols.append(target)
        df = self.drop_empty(self.df_raw[all_cols])
        return {
            "X" : df[cols].drop(columns=target),
            "Y" : df[target]
            }
    
    def interference_valuation_data(self, target = None):
        if target == None:
            target = self.target
        df = self.SOP_data.copy()
        df = df.drop(columns=[
            "ID_PROJ", 
            "ID_VOLUNT",
            "IDADE",
            'ALTURA',
            # Metrics
                #T2
            'IMC2_CLASS',
            'CC_T2_CLAS',
            'RCQ_T2_CLAS',
            'RCE_T2_CLAS',
            'CP_T2_CLAS',
            'CB_T2_CLAS',
            'CB_T2_ADEQUA',
                #T1
            'IMC1_CLASS',
            'CC_T1_CLAS',
            'RCQ_T1_CLAS',
            'RCE_T1_CLAS',
            'CP_T1_CLAS',
            'CB_T1_CLAS',
            'CB_T1_ADEQUA',
            # Diet
                #T2
            'PESO_T2','VCT_T2','CHO_G_T2',
            'CHO_%_T2','PTN_G_T2','PTN_%_T2',
            'LIP_G_T2','LIP_%_T2','COLESTEROL_T2',
            'ADEQ_COLES_T2','G_SAT_T2','ADEQ_G_SAT_T2',
                #T2
            'PESO_T1','VCT_T1','CHO_G_T1','CHO_%_T1',
            'PTN_G_T1','PTN_%_T1','LIP_G_T1',
            'LIP_%_T1','COLESTEROL_T1','ADEQ_COLES_T1',
            'G_SAT_T1','ADEQ_G_SAT_T1',
            #Mental
                #T1
            'T1_PONT_TOTAL','T1_PONT_A','T1_PONT_D',
                #T2
            'T2_PONT_TOTAL','T2_PONT_A','T2_PONT_D',
            # Other
            # 'IMC_T1',
            # 'CC_T1',
            # 'CQ_T1',
            # 'RCQ_T1',
            # 'RCE_T1',
            # 'CP_T1',
            # 'CB_T1',
            ])
        q_cols ={}
        for i in range(1,24):
            q_cols[f'Q_{i}_T1'] = f'Q_{i}_T2'
        q_cols["TOTAl_T1"] = "TOTAl_T2"
        for key_1, key_2 in q_cols.items():
            df[key_1] = df[key_1].replace(999, 0.0)
            mask = df[key_1] != 999
            df.loc[mask & (df[key_2] == 999), key_2] = df.loc[mask, key_1]
        return {
            "Y": df[target],
            "X": df.drop(columns=self.targets)
        }
        
    
    def instantiate_models(self):
        self.models ={
            'lr_clf' : LogisticRegression(random_state=0),
            'rf_clf' : RandomForestClassifier(random_state=0),
            'gb_clf' : GaussianNB(),
            'knn_clf' : neighbors.KNeighborsClassifier(),
            'svm_clf' : svm.SVC(kernel='linear', probability = True),
            'svm_rbf_clf' : svm.SVC(kernel='rbf', probability = True),
            'svm_sig_clf' : svm.SVC(kernel='sigmoid', probability = True),
            'xgb_clf' : XGBClassifier(),
            'gbc_clf' : GradientBoostingClassifier(random_state=0),
            'dt_clf' : DecisionTreeClassifier(random_state=0)}
    
    def rank_features(self, data : dict):
        X = data['X']
        y = data['Y']
        ridge = RidgeCV(alphas=np.logspace(-6, 6, num=5)).fit(X, y)

        importance = np.abs(ridge.coef_)
        feature_names = np.array(X.columns)

        # Sort the features by their importance
        indices = np.argsort(importance)[::-1]
        top_feature_indices = indices[:]
        try:
            top_features = feature_names[top_feature_indices[:23]]
            top_importance = importance[top_feature_indices[:23]]
        except:
            top_features = feature_names[top_feature_indices]
            top_importance = importance[top_feature_indices]
        plt.clf()
        plt.barh(top_features, top_importance)
        plt.xlabel('Importance')
        plt.title("Feature Importances via Coefficients")
        plt.gca().invert_yaxis()  # Invert y-axis to have the most important feature on top
        plt.tight_layout()  # Adjust layout to prevent clipping of labels
        plt.savefig(f'results/{self.target}/Feature_Importance.png')
        return top_features.tolist()
    
    def select_best_model_ridge(self, data):
        best_models = {}
        data
        X_train,X_test,Y_train,Y_test = train_test_split(data['X'],data['Y'], test_size=0.3, random_state=0)
        features_rank = self.rank_features(data)

        best_model = {'Name': None, 'AUC':0.0, 'variables': []}
        
        for N in range(4,len(data['X'].columns)):
            self.instantiate_models()
            for model_name, model in self.models.items():
                trained = model.fit(X_train[features_rank[:N]],Y_train)
                score = trained.score(X_test[features_rank[:N]], Y_test)
                y_pred_proba = model.predict_proba(X_test[features_rank[:N]])
                y_pred = trained.predict(X_test[features_rank[:N]])
                auc = roc_auc_score(Y_test, y_pred_proba[:, 1])
                if auc > best_model['AUC']:
                    best_model['Name'] = model_name
                    best_model['AUC'] = auc
                    best_model['variables'] = features_rank[:N]
                    best_model['Model'] = model
                    best_model['target'] = self.target
        self.confusion_matrix(best_model, data)
        # self.plot_auc_from_dict(best_model, data
        #                          )
        self.plot_auc_from_dict(best_model, {"X": X_test, "Y" : Y_test} )
        self.plot_metrics(best_model,X_test,Y_test)
        return best_model
    
    @staticmethod
    def confusion_matrix(best_model, data):
        model = best_model['Model']
        features = best_model['variables']
        name = best_model['Name']
        target = best_model['target']
        y_pred = model.predict(data['X'][features])
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            confusion_matrix(data['Y'], y_pred), 
            annot=True, 
            fmt='d', 
            cmap='Blues', 
            xticklabels=np.unique(data['Y']), 
            yticklabels=np.unique(data['Y'])
            )
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(f'Confusion Matrix for {name}')
        plt.savefig(f'results/{target}/Confusion_Matrix_{name}.png')
    
    def fix_smart(self):
        cols = ["SEGUIU_METAS_1","SEGUIU_METAS2","SEGUIU_METAS3"]
        for col in cols:
            self.df_raw[col]=self.df_raw[col].apply(lambda x: 0.0 if (x == 999) or np.isnan(x) else x )

    def fix_mental(self):
        mental_t2 = ['T2_PONT_TOTAL','T2_PONT_A','T2_PONT_D']
        mental_t1 = [x.replace("T2","T1") for x in mental_t2]
        for key_1, key_2 in zip(mental_t1, mental_t2):
            mask = self.df_raw[key_1] != 999
            self.df_raw.loc[mask & (self.df_raw[key_2] == 999), key_2] = self.df_raw.loc[mask, key_1]

    def fix_diet(self):
        diet_t2 = [
            'VCT_T2','CHO_G_T2','CHO_%_T2','PTN_G_T2',
            'PTN_%_T2','LIP_G_T2','LIP_%_T2','COLESTEROL_T2',
            'ADEQ_COLES_T2','G_SAT_T2','ADEQ_G_SAT_T2'
        ]
        diet_t1 = [x.replace("T2","T1") for x in diet_t2]
        for key_1, key_2 in zip(diet_t1, diet_t2):
            mask = self.df_raw[key_1] != 999
            self.df_raw.loc[mask & (self.df_raw[key_2] == 999), key_2] = self.df_raw.loc[mask, key_1]

    def delta_targets(self):
        pairs = {
            'IMC_T2' : 'IMC_T1',
            'CC_T2' : 'CC_T1',
            'CQ_T2' : 'CQ_T1',
            'RCQ_T2' : 'RCQ_T1',
            'RCE_T2' : 'RCE_T1',
            'CP_T2' : 'CP_T1',
            'CB_T2' : 'CB_T1',
        }
        new_target_cols = []
        for t_2, t_1 in pairs.items():
            new_col = t_2.replace("_T2","_D")
            new_target_cols.append(new_col)
            self.df_raw[new_col] = self.df_raw[t_2] - self.df_raw[t_1]
            self.df_raw[new_col] = self.df_raw[new_col].apply(lambda x : 1 if x < 0 else 0 )
            self.df_raw = self.df_raw.drop(columns=[t_2,t_1])
        self.targets = new_target_cols
        return new_target_cols
    
    @staticmethod
    def plot_auc_from_dict(model, data_dict):
        X_test = data_dict['X']
        y_test = data_dict['Y']
        name = model['Name']
        target = model['target']
        # Make predictions on the test set
        try:
            y_prob = model['Model'].predict_proba(X_test[model["variables"]])[:, 1]
        except:
            y_prob = model['Model']._predict_proba_lr(X_test[model["variables"]])[:, 1]
        # Compute ROC curve and AUC
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        # Plot ROC curve
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'Receiver Operating Characteristic (ROC) for {name} with target {target}')
        plt.legend(loc="lower right")
        plt.savefig(f'results/{target}/AUC_{name}_{target}.png')
    
    def save_histograms(self,data,columns):
        target = self.target
        data = data["X"]
        for column in columns:
            plt.figure(figsize=(8, 6))
            data[column].hist(grid=False, edgecolor='black')
            plt.title(f'Histogram of {column}')
            plt.xlabel(column)
            plt.ylabel('Frequency')
            plt.savefig(f'results/{target}/{column}_histogram.png')
            plt.close()
    
    
    def save_histograms_c1(self,data,columns):
        all_cols = columns.copy()
        all_cols.append('ID_VOLUNT')
        data = self.drop_empty(data[all_cols])
        data_sop = data[data['ID_VOLUNT'] == 0]
        data_nosop = data[data['ID_VOLUNT'] == 1]
        target =  self.target
        for column in columns:
            plt.figure(figsize=(8, 6))
            data_sop[column].hist(grid=False, edgecolor='black')
            plt.title(f'Histogram of {column} for control')
            plt.xlabel(column)
            plt.ylabel('Frequency')
            plt.savefig(f'results/{target}/{column}_histogram_sop.png')
            plt.close()
        for column in columns:
            plt.figure(figsize=(8, 6))
            data_nosop[column].hist(grid=False, edgecolor='black')
            plt.title(f'Histogram of {column} for SOP')
            plt.xlabel(column)
            plt.ylabel('Frequency')
            plt.savefig(f'results/{target}/{column}_histogram_no_sop.png')
            plt.close()

    def plot_metrics(self,best_model, X_test, y_test):
        model = best_model["Model"]
        name = best_model["Name"]
        y_pred_proba = model.predict_proba(X_test[best_model["variables"]])
        y_pred = model.predict(X_test[best_model["variables"]])
        
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba[:, 1])
        logloss = log_loss(y_test, y_pred_proba)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        
        
        metrics = {'F1 Score': f1, 'AUC': auc, 'Recall': recall, 'Precision': precision}
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(metrics.keys(), metrics.values(), color='skyblue')
        plt.xlabel('Metrics')
        plt.ylabel('Value')
        plt.title('Model Metrics')
        plt.xticks(rotation=45)
        
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, round(yval, 4), ha='center', va='bottom')
        plt.savefig(f'results/{self.target}/Metrics_{name}_{self.target}.png')


if __name__ == "__main__":
    run = Process(target="ID_VOLUNT")
    run.instantiate_models()
    model = run.select_best_model_ridge(run.condition_quest_data)
    run.save_histograms_c1(run.df_raw,model['variables'])
    for target in ['IMC_D', 'CC_D', 'CQ_D', 'RCQ_D', 'RCE_D', 'CP_D', 'CB_D']:
        print(f"Gerando para {target}")
        run = Process(target=target)
        d_targets = run.delta_targets()
        run.instantiate_models()
        model = run.select_best_model_ridge(run.interference_valuation_data(target))
        run.save_histograms(run.interference_valuation_data(target),model['variables'])
    