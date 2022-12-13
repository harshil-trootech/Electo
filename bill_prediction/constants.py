LEGISLATOR_POLICY_AREA_FILE_PATH = 'bill_prediction/outputs/files/legislator_policy_area.csv'
LEGISLATOR_SUBJECTS_FILE_PATH = 'bill_prediction/outputs/files/legislator_subjects.csv'
HOUSE_POLICY_AREA_FILE_PATH = 'bill_prediction/outputs/files/house_policy_area.csv'
HOUSE_SUBJECTS_FILE_PATH = 'bill_prediction/outputs/files/house_subjects.csv'
SENATE_POLICY_AREA_FILE_PATH = 'bill_prediction/outputs/files/senate_policy_area.csv'
SENATE_SUBJECTS_FILE_PATH = 'bill_prediction/outputs/files/senate_subjects.csv'
CURRENT_GOVERNMENT_DETAILS_FILE_PATH = 'bill_prediction/outputs/files/current_government_details.json'
HOUSE_FEATURE_CACHE_FILE_PATH = 'bill_prediction/outputs/files/house_features.csv'
SENATE_FEATURE_CACHE_FILE_PATH = 'bill_prediction/outputs/files/senate_features.csv'
PREDICTION_FILE_PATH = 'bill_prediction/outputs/results/'
HOUSE_MODEL_PATH = 'bill_prediction/outputs/models/house/'
SENATE_MODEL_PATH = 'bill_prediction/outputs/models/senate/'
HOUSE_PASS_STATUS_LIST = ['ENACTED:SIGNED', 'PASS_OVER:HOUSE', 'PASS_BACK:SENATE',
                          'PASSED:BILL', 'PASSED:SIMPLERES', 'PASSED:CONCURRENTRES']
HOUSE_FAIL_STATUS_LIST = ['FAIL:ORIGINATING:HOUSE', 'FAIL:SECOND:HOUSE']
SENATE_PASS_STATUS_LIST = ['ENACTED:SIGNED', 'PASS_OVER:SENATE', 'PASS_BACK:HOUSE',
                           'PASSED:BILL', 'PASSED:SIMPLERES', 'PASSED:CONCURRENTRES']
SENATE_FAIL_STATUS_LIST = ['FAIL:ORIGINATING:SENATE', 'FAIL:SECOND:SENATE']
