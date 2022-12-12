import os
from django.core.management.base import BaseCommand, CommandError
from bill_prediction.helper.data_loader import DataLoader, FeatureExtractor
from bill.models import Bill
from bill_prediction.constants import *
import xgboost as xgb
import pandas as pd
import numpy as np
import joblib
from tqdm import tqdm


class Command(BaseCommand):
    def handle(self, *args, **options):
        feature_extractor = FeatureExtractor()
        house_model_names = os.listdir(HOUSE_MODEL_PATH)
        senate_model_names = os.listdir(SENATE_MODEL_PATH)
        house_model_list = []
        senate_model_list = []

        print("...Loading house models...")
        for model_name in house_model_names:
            house_model_list.append(joblib.load(HOUSE_MODEL_PATH + model_name))
        print("...Loading senate models...")
        for model_name in senate_model_names:
            senate_model_list.append(joblib.load(SENATE_MODEL_PATH + model_name))

        bills = Bill.objects.filter(status__in=['ENACTED:SIGNED', 'PASS_OVER:HOUSE', 'PASS_OVER:SENATE',
                                                'PASS_BACK:HOUSE', 'PASS_BACK:SENATE', 'FAIL:ORIGINATING:HOUSE',
                                                'FAIL:ORIGINATING:SENATE', 'FAIL:SECOND:HOUSE', 'FAIL:SECOND:SENATE'],
                                    bill_id__endswith='117')

        result = []
        print("...Generating result...")
        for bill in tqdm(bills):
            house_probs, senate_probs = [], []
            x_house = feature_extractor.get_features(bill, chamber='house', get_x=True)
            x_senate = feature_extractor.get_features(bill, chamber='senate', get_x=True)
            for model in house_model_list:
                house_probs.append(model.predict_proba([x_house])[0][1])
            house_probability = sum(house_probs)/len(house_probs)
            for model in senate_model_list:
                senate_probs.append(model.predict_proba([x_senate])[0][1])
            senate_probability = sum(senate_probs)/len(senate_probs)

            result.append({'id': bill.id,
                           'bill_id': bill.bill_id,
                           'status': bill.status,
                           'policy': bill.policy_area,
                           'House': house_probability,
                           'Senate': senate_probability,
                           'Overall': (house_probability+senate_probability)/2
                           })

        df = pd.DataFrame(result)
        print("saving file at bill_prediction/outputs/results/predictions.csv")
        df.to_csv('bill_prediction/outputs/results/predictions.csv', index=False)

        df = pd.read_csv('bill_prediction/outputs/results/predictions.csv')

        print("...Analysis on current congress bills...")

        df['house_passed'] = df['House'].apply(lambda x: 1 if x > 0.65 else 0)
        df['senate_passed'] = df['Senate'].apply(lambda x: 1 if x > 0.65 else 0)

        result = {'house': {'pass': [], 'fail': []}, 'senate': {'pass': [], 'fail': []}}

        house_passed_df = df[df['status'].isin(HOUSE_PASS_STATUS_LIST)].copy()
        result['house']['pass'].append(house_passed_df[house_passed_df['house_passed'] == 0]['house_passed'].count())
        result['house']['pass'].append(house_passed_df[house_passed_df['house_passed'] == 1]['house_passed'].count())

        house_failed_df = df[df['status'].isin(HOUSE_FAIL_STATUS_LIST)].copy()
        result['house']['fail'].append(house_failed_df[house_failed_df['house_passed'] == 0]['house_passed'].count())
        result['house']['fail'].append(house_failed_df[house_failed_df['house_passed'] == 1]['house_passed'].count())

        senate_passed_df = df[df['status'].isin(SENATE_PASS_STATUS_LIST)].copy()
        result['senate']['pass'].append(senate_passed_df[senate_passed_df['senate_passed'] == 0]['house_passed'].count())
        result['senate']['pass'].append(senate_passed_df[senate_passed_df['senate_passed'] == 1]['house_passed'].count())

        senate_failed_df = df[df['status'].isin(SENATE_FAIL_STATUS_LIST)].copy()
        result['senate']['fail'].append(senate_failed_df[senate_failed_df['senate_passed'] == 0]['house_passed'].count())
        result['senate']['fail'].append(senate_failed_df[senate_failed_df['senate_passed'] == 1]['house_passed'].count())

        print("House confusion matrix:")
        print(f"{result['house']['fail']}\n{result['house']['pass']}")
        print("Accuracy for house:", (result['house']['pass'][1]+result['house']['fail'][0])/(np.sum(result['house']['pass']+result['house']['fail'])))

        print("\nSenate confusion matrix:")
        print(f"{result['senate']['fail']}\n{result['senate']['pass']}")
        print("Accuracy for senate:", (result['senate']['pass'][1] + result['senate']['fail'][0]) / (np.sum(result['senate']['pass'] + result['senate']['fail'])))
