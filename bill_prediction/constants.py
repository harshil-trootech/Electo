LEGISLATOR_POLICY_AREA_FILE_PATH = 'bill_prediction/outputs/legislator_policy_area.csv'
LEGISLATOR_SUBJECTS_FILE_PATH = 'bill_prediction/outputs/legislator_subjects.csv'
HOUSE_POLICY_AREA_FILE_PATH = 'bill_prediction/outputs/house_policy_area.csv'
HOUSE_SUBJECTS_FILE_PATH = 'bill_prediction/outputs/house_subjects.csv'
SENATE_POLICY_AREA_FILE_PATH = 'bill_prediction/outputs/senate_policy_area.csv'
SENATE_SUBJECTS_FILE_PATH = 'bill_prediction/outputs/senate_subjects.csv'
CURRENT_GOVERNMENT_DETAILS_FILE_PATH = 'bill_prediction/outputs/current_government_details.json'
HOUSE_XGB_MODEL_PATH = 'bill_prediction/outputs/house_xgb.json'
HOUSE_RF_MODEL_PATH = 'bill_prediction/outputs/house_logistic.joblib'
SENATE_XGB_MODEL_PATH = 'bill_prediction/outputs/senate_xgb.json'
SENATE_RF_MODEL_PATH = 'bill_prediction/outputs/senate_logistic.joblib'
HOUSE_PASS_STATUS_LIST = ['PASS_OVER:HOUSE']
HOUSE_FAIL_STATUS_LIST = ['PASS_BACK:HOUSE', 'FAIL:ORIGINATING:HOUSE', 'FAIL:SECOND:HOUSE', 'PROV_KILL:CLOTUREFAILED',
                          'PROV_KILL:PINGPONGFAIL', 'PROV_KILL:SUSPENSIONFAILED', 'PROV_KILL:VETO']
SENATE_PASS_STATUS_LIST = ['PASS_OVER:SENATE']
SENATE_FAIL_STATUS_LIST = ['PASS_BACK:SENATE', 'FAIL:ORIGINATING:SENATE', 'FAIL:SECOND:SENATE',
                           'PROV_KILL:CLOTUREFAILED', 'PROV_KILL:PINGPONGFAIL', 'PROV_KILL:SUSPENSIONFAILED',
                           'PROV_KILL:VETO']
# PASS_STATUS_LIST = ['ENACTED:SIGNED']
# FAIL_STATUS_LIST = ['ENACTED:VETO_OVERRIDE', 'FAIL:ORIGINATING:HOUSE', 'FAIL:ORIGINATING:SENATE',
#                     'FAIL:SECOND:HOUSE', 'FAIL:SECOND:SENATE']
PREDICTION_STATUS_LIST = ['INTRODUCED']
