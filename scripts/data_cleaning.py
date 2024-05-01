import pandas as pd
import tabulate
import logging
class Cleaner:
    def __init__(self, path : str = "data/raw/survey.xlsx") -> None:
        self.logger: logging.Logger = logging.getLogger(__name__)
        try:
            self.df_raw = pd.read_excel(path, header = 1)
            self.logger.info('Raw data secured.')
        except FileNotFoundError:
            raise( "No survey file in folder")
        self.col_id = ['ID_PROJ','ID_VOLUNT','INICIAIS','IDADE','ALTURA']
        self.smart_id = ['SEGUIU_METAS_1','SEGUIU_METAS2','SEGUIU_METAS3']
        self.T1_id = {
            'Mental' : {
                'T1_PONT_TOTAL','T1_PONT_A','T1_PONT_D'
                }, 
            'Avaliacao':{
                'PESO_T1','IMC_T1','IMC1_CLASS','CC_T1',
                'CC_T1_CLAS','CQ_T1','RCQ_T1','RCQ_T1_CLAS',
                'RCE_T1','RCE_T1_CLAS','CP_T1','CP_T1_CLAS',
                'CB_T1','CB_T1_ADEQUA','CB_T1_CLAS'
                }, 
            'Dietetico':{
                'VCT_T1','CHO_G_T1','CHO_%_T1''PTN_G_T1',
                'PTN_%_T1','LIP_G_T1','LIP_%_T1','COLESTEROL_T1',
                'ADEQ_COLES_T1','G_SAT_T1','ADEQ_G_SAT_T1'
                },
            'Questionario':{
                'Q_1_T1','Q_2_T1','Q_3_T1','Q_4_T1','Q_5_T1',
                'Q_6_T1','Q_7_T1','Q_8_T1','Q_9_T1','Q_10_T1',
                'Q_11_T1','Q_12_T1','Q_13_T1','Q_14_T1','Q_15_T1',
                'Q_16_T1','Q_17_T1','Q_18_T1','Q_19_T1','Q_20_T1',
                'Q_21_T1','Q_22_T1','Q_23_T1','TOTAl_T1'
            }
        }
        self.T2_id = {
            'Mental' : {
                'T2_PONT_TOTAL','T2_PONT_A','T2_PONT_D'
                }, 
            'Avaliacao':{
                'PESO_T2','IMC_T2','IMC2_CLASS','CC_T2',
                'CC_T2_CLAS','CQ_T2','RCQ_T2','RCQ_T2_CLAS',
                'RCE_T2','RCE_T2_CLAS','CP_T2','CP_T2_CLAS',
                'CB_T2','CB_T2_ADEQUA','CB_T2_CLAS'
                }, 
            'Dietetico':{
                'VCT_T2','CHO_G_T2','CHO_%_T2','PTN_G_T2',
                'PTN_%_T2','LIP_G_T2','LIP_%_T2','COLESTEROL_T2',
                'ADEQ_COLES_T2','G_SAT_T2','ADEQ_G_SAT_T2'
                },
            'Questionario':{
                'Q_1_T2','Q_2_T2','Q_3_T2','Q_4_T2','Q_5_T2',
                'Q_6_T2','Q_7_T2','Q_8_T2','Q_9_T2','Q_10_T2',
                'Q_11_T2','Q_12_T2','Q_13_T2','Q_14_T2','Q_15_T2',
                'Q_16_T2','Q_17_T2','Q_18_T2','Q_19_T2','Q_20_T2',
                'Q_21_T2','Q_22_T2','Q_23_T2','TOTAl_T2'
            }
        }
    def fix_smart(self)->None:
        # FIXXXXXX
        self.logger.info('Remap Smart survey results to numerical scale')
        self.df_raw[self.smart_id] = self.df_raw[self.smart_id].map({1:1,2:0.5,0:0})
        print(tabulate(self.df_raw[self.smart_id],headers=self.smart_id, tablefmt="grid"))

if __name__ == '__main__':
    Cleaner().fix_smart()
    