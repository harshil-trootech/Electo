from django.core.management.base import BaseCommand, CommandError
from bill_prediction.helper.data_loader import DataLoader, FeatureExtractor
from bill.models import Bill
from bill_prediction.constants import *
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
from xgboost import XGBClassifier
import joblib
import shutil
import os


class Command(BaseCommand):
    def add_arguments(self, parser):
        parser.add_argument('--use_stats_cache', action='store_true', help='Use cached statistics file to generate features')
        parser.add_argument('--use_feature_cache', action='store_true', help='Use cached features file to train model')

    def train_ml_model(self, x_train, y_train, x_test, y_test, save_path, chamber):
        count = len(os.listdir(save_path)) + 1
        if chamber == 'house':
            model = LogisticRegression(max_iter=500)
        else:
            # model = DecisionTreeClassifier(min_samples_split=2, random_state=4)
            # model = GradientBoostingClassifier(n_estimators=50, max_features=3)
            model = LogisticRegression(max_iter=500, fit_intercept=True)
        model.fit(x_train, y_train)
        score = model.score(x_test, y_test)
        joblib.dump(model, f"{save_path}model{count}.joblib")
        return score

    def handle(self, *args, **options):
        if options['use_stats_cache']:
            self.stdout.write(self.style.SUCCESS("...Using cached file..."))
        else:
            self.stdout.write(self.style.SUCCESS("...Generating statistics files..."))
            DataLoader.generate_statistics_files()

        if options['use_feature_cache']:
            house_features_df = pd.read_csv('bill_prediction/outputs/house_features.csv')
            senate_features_df = pd.read_csv('bill_prediction/outputs/senate_features.csv')
        else:
            feature_extractor = FeatureExtractor()
            # Process to collect features and train model for the house
            print("...Collecting features of the house related bills...")
            house_bills = Bill.objects.filter(status__in=HOUSE_PASS_STATUS_LIST+HOUSE_FAIL_STATUS_LIST,
                                              introduced_at__year__gte=2014).exclude(bill_id__endswith='117')
            house_amendments = Bill.objects.filter(status__in=['pass', 'fail'], bill_id__startswith='h')
            # house_bills = house_related_bills | house_amendments
            house_bill_features = []
            for bill in tqdm(house_bills):
                house_bill_features.append(feature_extractor.get_features(bill, chamber='house'))

            house_features_df = pd.DataFrame(house_bill_features)
            house_features_df.to_csv('bill_prediction/outputs/house_features.csv', index=False)

            # Process to collect features and train model for the senate
            print("...Collecting features of the senate related bills...")
            senate_bills = Bill.objects.filter(status__in=SENATE_PASS_STATUS_LIST + SENATE_FAIL_STATUS_LIST,
                                               introduced_at__year__gte=2014).exclude(bill_id__endswith='117')
            # senate_amendments = Bill.objects.filter(status__in=['pass', 'fail'], bill_id__startswith='s')
            # senate_bills = senate_related_bills | senate_amendments
            senate_bill_features = []
            for bill in tqdm(senate_bills):
                senate_bill_features.append(feature_extractor.get_features(bill, chamber='senate'))

            senate_features_df = pd.DataFrame(senate_bill_features)
            senate_features_df.to_csv('bill_prediction/outputs/senate_features.csv', index=False)

        # Training models for house related bills
        print("...Training models for house related bills...")
        X = house_features_df.iloc[:, :-1]
        y = house_features_df.iloc[:, -1]
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
        x_test, y_test = x_test.values, y_test.values
        combined_df = pd.concat([x_train, y_train], axis=1)
        house_failed_bills = combined_df[combined_df.iloc[:, -1] == 0]
        house_passed_bills = combined_df[combined_df.iloc[:, -1] == 1].sample(frac=1)
        split_size = house_passed_bills.shape[0] // house_failed_bills.shape[0]
        passed_bills_bins = np.array_split(house_passed_bills, split_size)
        shutil.rmtree(HOUSE_MODEL_PATH, ignore_errors=True)
        os.makedirs(HOUSE_MODEL_PATH)
        scores = []
        for passed_bill in tqdm(passed_bills_bins):
            data = pd.concat([house_failed_bills, passed_bill]).sample(frac=1)
            x_train = data.iloc[:, :-1].values
            y_train = data.iloc[:, -1].values
            scores.append(self.train_ml_model(x_train, y_train, x_test, y_test, HOUSE_MODEL_PATH, 'house'))
        print("House score:", np.mean(scores))

        # Training models for senate related bills
        print("\n...Training models for senate related bills...")
        X = senate_features_df.iloc[:, :-1]
        y = senate_features_df.iloc[:, -1]
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
        x_test, y_test = x_test.values, y_test.values
        combined_df = pd.concat([x_train, y_train], axis=1)
        senate_failed_bills = combined_df[combined_df.iloc[:, -1] == 0]
        senate_passed_bills = combined_df[combined_df.iloc[:, -1] == 1].sample(frac=1)
        split_size = senate_passed_bills.shape[0] // senate_failed_bills.shape[0]
        passed_bills_bins = np.array_split(senate_passed_bills, split_size)
        shutil.rmtree(SENATE_MODEL_PATH, ignore_errors=True)
        os.makedirs(SENATE_MODEL_PATH)
        scores = []
        for passed_bill in tqdm(passed_bills_bins):
            data = pd.concat([senate_failed_bills, passed_bill]).sample(frac=1)
            x_train = data.iloc[:, :-1].values
            y_train = data.iloc[:, -1].values
            scores.append(self.train_ml_model(x_train, y_train, x_test, y_test, SENATE_MODEL_PATH, 'senate'))
        print("Senate score:", np.mean(scores))
