from django.core.management.base import BaseCommand, CommandError
from bill_prediction.helper.data_loader import DataLoader, FeatureExtractor
from bill.models import Bill
from bill_prediction.constants import *
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
from xgboost import XGBClassifier
import joblib


class Command(BaseCommand):
    def add_arguments(self, parser):
        parser.add_argument('--use_stats_cache', action='store_true', help='Use cached statistics file to generate features')
        parser.add_argument('--use_feature_cache', action='store_true', help='Use cached features file to train model')

    def train_xgb_model(self, df, save_path):
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
        xgb_model = XGBClassifier(booster='gblinear', n_estimators=25, learning_rate=0.3)  # , max_depth=5, min_samples_split=3)
        # xgb_model = GradientBoostingClassifier(n_estimators=151)
        xgb_model.fit(X_train, y_train)
        print("Confusion metrics")
        y_pred = xgb_model.predict_proba(X_test)[:, 1]
        y_pred = [0 if y < 0.5 else 1 for y in y_pred]
        print(confusion_matrix(y_test, y_pred))
        self.stdout.write(self.style.SUCCESS(f"Training accuracy: {xgb_model.score(X_train, y_train)}"))
        self.stdout.write(self.style.SUCCESS(f"Testing accuracy: {xgb_model.score(X_test, y_test)}"))
        self.stdout.write(self.style.SUCCESS(f"Saving model at {save_path}"))
        xgb_model.save_model(save_path)
        # joblib.dump(xgb_model, save_path)

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
            house_related_bills = Bill.objects.filter(status__in=HOUSE_PASS_STATUS_LIST+HOUSE_FAIL_STATUS_LIST,
                                                      introduced_at__year__gte=2015).exclude(bill_id__endswith='117')
            house_amendments = Bill.objects.filter(status__in=['pass', 'fail'], bill_id__startswith='h')
            house_bills = house_related_bills | house_amendments
            house_bill_features = []
            for bill in tqdm(house_bills):
                house_bill_features.append(feature_extractor.get_features(bill, chamber='house'))

            house_bill_df = pd.DataFrame(house_bill_features)
            house_failed_bills = house_bill_df[house_bill_df['label'] == 0]
            house_passed_bills = house_bill_df[house_bill_df['label'] == 1].sample(house_failed_bills.shape[0])
            # house_passed_bills = house_bill_df[house_bill_df['label'] == 1]
            house_features_df = pd.concat([house_failed_bills, house_passed_bills])
            house_features_df = house_features_df.sample(frac=1)
            house_features_df.reset_index(drop=True, inplace=True)
            house_features_df.to_csv('bill_prediction/outputs/house_features.csv', index=False)

            # Process to collect features and train model for the senate
            print("...Collecting features of the senate related bills...")
            senate_related_bills = Bill.objects.filter(status__in=SENATE_PASS_STATUS_LIST + SENATE_FAIL_STATUS_LIST,
                                                       introduced_at__year__gte=2015).exclude(bill_id__endswith='117')
            senate_amendments = Bill.objects.filter(status__in=['pass', 'fail'], bill_id__startswith='s')
            senate_bills = senate_related_bills | senate_amendments
            senate_bill_features = []
            for bill in tqdm(senate_bills):
                senate_bill_features.append(feature_extractor.get_features(bill, chamber='senate'))

            senate_bill_df = pd.DataFrame(senate_bill_features)
            senate_failed_bills = senate_bill_df[senate_bill_df['label'] == 0]
            senate_passed_bills = senate_bill_df[senate_bill_df['label'] == 1].sample(senate_failed_bills.shape[0])
            # senate_passed_bills = senate_bill_df[senate_bill_df['label'] == 1]
            senate_features_df = pd.concat([senate_failed_bills, senate_passed_bills])
            senate_features_df = senate_features_df.sample(frac=1)
            senate_features_df.reset_index(drop=True, inplace=True)
            senate_features_df.to_csv('bill_prediction/outputs/senate_features.csv', index=False)

        print("...Training XGB Classifier for house...")
        self.train_xgb_model(house_features_df, HOUSE_XGB_MODEL_PATH)
        print("...Training XGB Classifier for senate...")
        self.train_xgb_model(senate_features_df, SENATE_XGB_MODEL_PATH)

        # print("...Finding best parameters from the grid...")
        # n_fold = KFold(5)
        # param_grid = {
        #     'booster': ['gbtree', 'gblinear', 'dart'],
        #     'n_estimators': [25, 50, 100],
        #     'max_depth': [5, 7, 9, 11],
        #     'learning_rate': [0.1, 0.2, 0.3]
        # }
        # reg_alpha
        # reg_lambda
        # model = XGBClassifier()
        # cv = GridSearchCV(model, param_grid, cv=n_fold)
        # cv.fit(X, y)
        # print(f"Best score: {cv.best_score_}")
        # print(f"Best parameters: {cv.best_params_}")
        #
        X = house_features_df.iloc[:, :-1].values
        y = house_features_df.iloc[:, -1].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
        print("...Training logistic regression model...")
        lr_model_house = LogisticRegression(max_iter=1000, fit_intercept=False)
        lr_model_house.fit(X_train, y_train)
        print(lr_model_house.coef_)
        self.stdout.write(self.style.SUCCESS(f"Training accuracy: {lr_model_house.score(X_train, y_train)}"))
        self.stdout.write(self.style.SUCCESS(f"Testing accuracy: {lr_model_house.score(X_test, y_test)}"))
        print("Confusion metrics")
        y_pred = lr_model_house.predict_proba(X_test)[:, 1]
        y_pred = [0 if y < 0.5 else 1 for y in y_pred]
        print(confusion_matrix(y_test, y_pred))
        self.stdout.write(self.style.SUCCESS(f"Saving model at {PICKLE_MODEL_PATH}"))
        joblib.dump(lr_model_house, PICKLE_MODEL_PATH)
