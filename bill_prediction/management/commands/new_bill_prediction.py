import os
from django.core.management.base import BaseCommand, CommandError
from bill_prediction.helper.data_loader import DataLoader, FeatureExtractor
from bill.models import Bill
from bill_prediction.constants import *
from datetime import date
import pandas as pd
import numpy as np
import joblib
from tqdm import tqdm


class Command(BaseCommand):

    def convert_house_range(self, num):
        threshold = 0.5
        if num < threshold:
            old_range = threshold
            new_range = 0.5
            return (num * new_range)/old_range
        if num >= threshold:
            old_range = 1 - threshold
            new_range = 0.5
            return (((num - threshold) * new_range) / old_range) + 0.5

    def convert_senate_range(self, num):
        threshold = 0.55
        if num < threshold:
            old_range = threshold
            new_range = 0.5
            return (num * new_range)/old_range
        if num >= threshold:
            old_range = 1 - threshold
            new_range = 0.5
            return (((num - threshold) * new_range) / old_range) + 0.5

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

        bills = Bill.objects.filter(status__in=['PASS_OVER:HOUSE', 'PASS_OVER:SENATE', 'ENACTED:SIGNED', 'INTRODUCED'
                                                'FAIL:ORIGINATING:HOUSE', 'FAIL:ORIGINATING:SENATE',
                                                'FAIL:SECOND:HOUSE', 'FAIL:SECOND:SENATE', 'PASS_BACK:HOUSE',
                                                'PASS_BACK:SENATE', 'REFERRED'],
                                    bill_id__endswith='117', modified__date__gte=date(year=2022, month=12, day=13))\
                            .exclude(policy_area=None).order_by('-modified')

        result = []
        house_features, senate_features = [], []
        print("...Generating result...")
        for bill in tqdm(bills):
            house_probs, senate_probs = [], []
            x_house = feature_extractor.get_features(bill, chamber='house', get_x=True)
            house_features.append(x_house)
            x_senate = feature_extractor.get_features(bill, chamber='senate', get_x=True)
            senate_features.append(x_senate)
            for model in house_model_list:
                house_probs.append(model.predict_proba([x_house])[0][1])
            house_probability = sum(house_probs)/len(house_probs)
            house_probability = self.convert_house_range(house_probability)
            for model in senate_model_list:
                senate_probs.append(model.predict_proba([x_senate])[0][1])
            senate_probability = sum(senate_probs)/len(senate_probs)
            senate_probability = self.convert_senate_range(senate_probability)

            result.append({'id': bill.id,
                           'modified': bill.modified,
                           'bill_id': bill.bill_id,
                           'status': bill.status,
                           'policy': bill.policy_area,
                           'House': house_probability,
                           'Senate': senate_probability,
                           'Overall': (house_probability+senate_probability)/2
                           })

        current_df = pd.DataFrame(result)
        file_path = f"{PREDICTION_FILE_PATH}prediction_new.csv"
        print("saving file at "+file_path)
        current_df.to_csv(file_path, index=False)

        # generating file with different status that base
        old_base_df = pd.read_csv(f"{PREDICTION_FILE_PATH}prediction_base.csv")
        old_base_df.set_index('bill_id', inplace=True)

        new_base_df = pd.read_csv(f"{PREDICTION_FILE_PATH}prediction_base.csv")

        difference_list = []
        for bill in new_base_df.itertuples():
            difference = {}
            bill_id = bill.bill_id
            status = bill.status
            try:
                old_bill = old_base_df.loc[bill_id]
                if status == old_bill.status:
                    continue
            except KeyError:
                pass

            # Code to compare two status and generate file with different status

            break

