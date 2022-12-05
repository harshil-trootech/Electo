from django.core.management.base import BaseCommand, CommandError
from bill_prediction.helper.data_loader import DataLoader, FeatureExtractor
# from electo_premium.bill.models import Bill
# from electo_premium.bill_prediction.constants import *
# import pandas as pd
# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import train_test_split, KFold, GridSearchCV
# from sklearn.metrics import confusion_matrix
# from xgboost import XGBClassifier
# import pickle as pkl


class Command(BaseCommand):
    def add_arguments(self, parser):
        parser.add_argument('--use_cache', action='store_true', help='Use cached statistics file to generate features')

    def handle(self, *args, **options):

        if options['use_cache']:
            self.stdout.write(self.style.SUCCESS("...Using cached file..."))
        else:
            self.stdout.write(self.style.SUCCESS("...Generating statistics files..."))
            DataLoader.generate_statistics_files()

        # Process to collect features and train model
        # feature_extractor = FeatureExtractor()
        # print("...Collecting features of the bills...")
        # bills = Bill.objects.filter(status__in=PASS_STATUS_LIST+FAIL_STATUS_LIST)
        # bill_features = []
        #
        # for bill in bills:
        #     bill_features.append(feature_extractor.get_features(bill))
        #
        # bill_df = pd.DataFrame(bill_features)
        # failed_bills = bill_df[bill_df['label'] == 0]
        # passed_bills = bill_df[bill_df['label'] == 1].sample(failed_bills.shape[0])
        # # passed_bills = bill_df[bill_df['label'] == 1]
        # combined_df = pd.concat([failed_bills, passed_bills])
        # combined_df = combined_df.sample(frac=1)
        # combined_df.reset_index(drop=True, inplace=True)
        #
        # X = combined_df.iloc[:, :-1].values
        # y = combined_df.iloc[:, -1].values
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
        #
        # print("...Training XGB Classifier...")
        # xgb_model = XGBClassifier(booster='gblinear')
        # xgb_model.fit(X_train, y_train)
        # self.stdout.write(self.style.SUCCESS(f"Training accuracy: {xgb_model.score(X_train, y_train)}"))
        # self.stdout.write(self.style.SUCCESS(f"Testing accuracy: {xgb_model.score(X_test, y_test)}"))
        # print("Confusion metrics")
        # y_pred = xgb_model.predict_proba(X_test)[:, 1]
        # y_pred = [0 if y < 0.65 else 1 for y in y_pred]
        # cm = confusion_matrix(y_test, y_pred)
        # print(cm)
        # print("Recall for 0: ", cm[0][0] / cm[0, :].sum())
        # print("Recall for 1: ", cm[1][1] / cm[1, :].sum())
        # print("Precision for 0: ", cm[0][0] / cm[:, 0].sum())
        # print("Precision for 1: ", cm[1][1] / cm[:, 1].sum())
        # self.stdout.write(self.style.SUCCESS(f"Saving model at {PREDICTION_MODEL_PATH}"))
        # xgb_model.save_model(PREDICTION_MODEL_PATH)
        #
        # # print("...Finding best parameters from the grid...")
        # # n_fold = KFold(5)
        # # param_grid = {
        # #     'n_estimators': [25, 50, 100],
        # #     'max_depth': [5, 7, 9, 11],
        # #     'min_samples_split': [5, 7, 9, 11],
        # #     'learning_rate': [0.1, 0.2, 0.3]
        # # }
        # # model = XGBClassifier()
        # # cv = GridSearchCV(model, param_grid, cv=n_fold)
        # # cv.fit(X, y)
        # # print(f"Best score: {cv.best_score_}")
        # # print(f"Best parameters: {cv.best_params_}")
        #
        # print("...Training logistic regression model...")
        # lr_model = LogisticRegression(max_iter=1000, fit_intercept=False)
        # lr_model.fit(X_train, y_train)
        # print(lr_model.coef_)
        # self.stdout.write(self.style.SUCCESS(f"Training accuracy: {lr_model.score(X_train, y_train)}"))
        # self.stdout.write(self.style.SUCCESS(f"Testing accuracy: {lr_model.score(X_test, y_test)}"))
        # print("Confusion metrics")
        # y_pred = lr_model.predict_proba(X_test)[:, 1]
        # y_pred = [0 if y < 0.65 else 1 for y in y_pred]
        # cm = confusion_matrix(y_test, y_pred)
        # print(cm)
        # print("Recall for 0: ", cm[0][0] / cm[0, :].sum())
        # print("Recall for 1: ", cm[1][1] / cm[1, :].sum())
        # print("Precision for 0: ", cm[0][0] / cm[:, 0].sum())
        # print("Precision for 1: ", cm[1][1] / cm[:, 1].sum())
        # self.stdout.write(self.style.SUCCESS(f"Saving model at {PICKLE_MODEL_PATH}"))
        # pkl.dump(lr_model, open(PICKLE_MODEL_PATH, 'wb'))
