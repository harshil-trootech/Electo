import joblib
from rest_framework import viewsets, status
from rest_framework.response import Response
from rest_framework.decorators import action
from bill.models import *
from .constants import *
from .helper.data_loader import FeatureExtractor
from .serializers import *
import xgboost as xgb
import pickle as pkl


class BillViewset(viewsets.ReadOnlyModelViewSet):
    queryset = Bill.objects.all()
    serializer_class = BillSerializer

    @action(detail=True, methods=['GET'])
    def features(self, request, pk=None):
        bill = self.get_object()

        try:
            feature_extractor = FeatureExtractor()
        except Exception as e:
            return Response(str(e), status=400)

        house_features = feature_extractor.get_features(bill, chamber='house', get_X_dict=True)
        senate_features = feature_extractor.get_features(bill, chamber='senate', get_X_dict=True)

        return Response({'house': house_features, 'senate': senate_features}, status=200)

    @action(detail=True, methods=['GET'])
    def prediction(self, request, pk=None):
        bill = self.get_object()
        try:
            feature_extractor = FeatureExtractor()
        except Exception as e:
            return Response(str(e), status=400)
        try:
            # house_model = xgb.XGBClassifier()
            # senate_model = xgb.XGBClassifier()
            # house_model.load_model(HOUSE_XGB_MODEL_PATH)
            # senate_model.load_model(SENATE_XGB_MODEL_PATH)
            house_model = joblib.load(HOUSE_RF_MODEL_PATH)
            senate_model = joblib.load(SENATE_RF_MODEL_PATH)
        except xgb.core.XGBoostError:
            return Response("Failed to load prediction model. Make sure to run python manage.py train_model before using this API endpoint", status=400)
        # pickle_model = pkl.load(open(PICKLE_MODEL_PATH, 'rb'))

        x_house = feature_extractor.get_features(bill, chamber='house', get_x=True)
        x_senate = feature_extractor.get_features(bill, chamber='senate', get_x=True)
        house_probability = house_model.predict_proba([x_house])
        senate_probability = senate_model.predict_proba([x_senate])

        return Response({'House': house_probability,
                         "Senate": senate_probability},
                         # "Overall": (house_probability+senate_probability)/2},
                        status=200)
