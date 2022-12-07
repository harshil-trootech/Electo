from django.core.management.base import BaseCommand, CommandError
from bill_prediction.helper.data_loader import DataLoader, FeatureExtractor
from bill.models import Bill
from bill_prediction.constants import *
import xgboost as xgb
import pandas as pd
import joblib
from tqdm import tqdm


class Command(BaseCommand):
    def handle(self, *args, **options):
        feature_extractor = FeatureExtractor()
        house_model = xgb.XGBClassifier()
        senate_model = xgb.XGBClassifier()
        house_model.load_model(HOUSE_XGB_MODEL_PATH)
        senate_model.load_model(SENATE_XGB_MODEL_PATH)
        lr_model = joblib.load(PICKLE_MODEL_PATH)

        bills = Bill.objects.filter(status__in=['ENACTED:SIGNED', 'PASS_OVER:HOUSE', 'PASS_BACK:SENATE',
                                                'PASS_OVER:SENATE', 'PASS_BACK:HOUSE', 'PASS_BACK:HOUSE',
                                                'FAIL:ORIGINATING:HOUSE', 'FAIL:SECOND:HOUSE', 'PASS_BACK:SENATE',
                                                'FAIL:ORIGINATING:SENATE', 'FAIL:SECOND:SENATE'],
                                    bill_id__endswith='117')

        result = []
        for bill in tqdm(bills):
            x_house = feature_extractor.get_features(bill, chamber='house', get_x=True)
            x_senate = feature_extractor.get_features(bill, chamber='senate', get_x=True)
            house_probability = house_model.predict_proba([x_house])[0][1]
            house_probability_lr = lr_model.predict_proba([x_house])[0][1]
            senate_probability = senate_model.predict_proba([x_senate])[0][1]
            result.append({'id': bill.id,
                           'bill_id': bill.bill_id,
                           'status': bill.status,
                           'House': house_probability,
                           'House Logistic': house_probability_lr,
                           'Senate': senate_probability})

        df = pd.DataFrame(result)
        print("saving file at bill_prediction/outputs/results/result.csv")
        df.to_csv('bill_prediction/outputs/results/result.csv')
