LEGISLATOR_POLICY_AREA_FILE_PATH = 'bill_prediction/outputs/legislator_policy_area.csv'
LEGISLATOR_SUBJECTS_FILE_PATH = 'bill_prediction/outputs/legislator_subjects.csv'
HOUSE_POLICY_AREA_FILE_PATH = 'bill_prediction/outputs/house_policy_area.csv'
HOUSE_SUBJECTS_FILE_PATH = 'bill_prediction/outputs/house_subjects.csv'
SENATE_POLICY_AREA_FILE_PATH = 'bill_prediction/outputs/senate_policy_area.csv'
SENATE_SUBJECTS_FILE_PATH = 'bill_prediction/outputs/senate_subjects.csv'
CURRENT_GOVERNMENT_DETAILS_FILE_PATH = 'bill_prediction/outputs/current_government_details.json'
PREDICTION_MODEL_PATH = 'bill_prediction/outputs/predictor.json'
PICKLE_MODEL_PATH = 'bill_prediction/outputs/predictor.pkl'
RANDOM_FOREST_MODEL_PATH = 'bill_prediction/outputs/random_forest.joblib'
HOUSE_PASS_STATUS_LIST = ['PASS_OVER:HOUSE', 'CONFERENCE:PASSED:HOUSE']
HOUSE_FAIL_STATUS_LIST = ['PASS_BACK:HOUSE', 'FAIL:ORIGINATING:HOUSE', 'FAIL:SECOND:HOUSE']
SENATE_PASS_STATUS_LIST = ['PASS_OVER:SENATE', 'CONFERENCE:PASSED:SENATE']
SENATE_FAIL_STATUS_LIST = ['PASS_BACK:SENATE', 'FAIL:ORIGINATING:SENATE', 'FAIL:SECOND:SENATE']
PASS_STATUS_LIST = ['ENACTED:SIGNED']
FAIL_STATUS_LIST = ['ENACTED:VETO_OVERRIDE', 'FAIL:ORIGINATING:HOUSE', 'FAIL:ORIGINATING:SENATE',
                    'FAIL:SECOND:HOUSE', 'FAIL:SECOND:SENATE']
PREDICTION_STATUS_LIST = ['INTRODUCED']
