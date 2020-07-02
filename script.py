import pandas as pd
import gme as gme
testdata=pd.read_csv('/home/tisaac/var/tmp/itpde_estim_58.csv')
testdata.columns
gme_data = gme.EstimationData(data_frame = testdata,
                              imp_var_name = 'importer',
                              exp_var_name = 'exporter',
                              trade_var_name = 'trade',
                              year_var_name = 'year')
gme_model = gme.EstimationModel(estimation_data = gme_data,
                                lhs_var = 'trade',
                                rhs_var = ['DIST','FTA','LANG','CNTG'],
                                fixed_effects = ['importer', 'exporter'],
                                keep_years = [2013,2014,2015])
est = gme_model.estimate()
