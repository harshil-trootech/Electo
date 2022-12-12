LEGISLATOR_POLICY_AREA_FILE_PATH = 'bill_prediction/outputs/files/legislator_policy_area.csv'
LEGISLATOR_SUBJECTS_FILE_PATH = 'bill_prediction/outputs/files/legislator_subjects.csv'
HOUSE_POLICY_AREA_FILE_PATH = 'bill_prediction/outputs/files/house_policy_area.csv'
HOUSE_SUBJECTS_FILE_PATH = 'bill_prediction/outputs/files/house_subjects.csv'
SENATE_POLICY_AREA_FILE_PATH = 'bill_prediction/outputs/files/senate_policy_area.csv'
SENATE_SUBJECTS_FILE_PATH = 'bill_prediction/outputs/files/senate_subjects.csv'
CURRENT_GOVERNMENT_DETAILS_FILE_PATH = 'bill_prediction/outputs/files/current_government_details.json'
HOUSE_MODEL_PATH = 'bill_prediction/outputs/models/house/'
SENATE_MODEL_PATH = 'bill_prediction/outputs/models/senate/'
HOUSE_PASS_STATUS_LIST = ['ENACTED:SIGNED', 'PASS_OVER:HOUSE', 'PASS_BACK:SENATE']
HOUSE_FAIL_STATUS_LIST = ['PASS_BACK:HOUSE', 'FAIL:ORIGINATING:HOUSE', 'FAIL:SECOND:HOUSE']
SENATE_PASS_STATUS_LIST = ['ENACTED:SIGNED', 'PASS_OVER:SENATE', 'PASS_BACK:HOUSE']
SENATE_FAIL_STATUS_LIST = ['PASS_BACK:SENATE', 'FAIL:ORIGINATING:SENATE', 'FAIL:SECOND:SENATE']
# PASS_STATUS_LIST = ['ENACTED:SIGNED']
# FAIL_STATUS_LIST = ['ENACTED:VETO_OVERRIDE', 'FAIL:ORIGINATING:HOUSE', 'FAIL:ORIGINATING:SENATE',
#                     'FAIL:SECOND:HOUSE', 'FAIL:SECOND:SENATE']
PREDICTION_STATUS_LIST = ['INTRODUCED']
