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

        features = feature_extractor.get_features()

        return Response(features, status=200)

    @action(detail=True, methods=['GET'])
    def prediction(self, request, pk=None):
        bill = self.get_object()
        try:
            feature_extractor = FeatureExtractor()
        except Exception as e:
            return Response(str(e), status=400)
        try:
            xgb_model = xgb.XGBClassifier()
            xgb_model.load_model(PREDICTION_MODEL_PATH)
        except xgb.core.XGBoostError:
            return Response("Failed to load prediction model. Make sure to run python manage.py train_model before using this API endpoint", status=400)
        pickle_model = pkl.load(open(PICKLE_MODEL_PATH, 'rb'))

        x = feature_extractor.get_features(bill, get_x=True)
        prediction1 = xgb_model.predict_proba([x])[0][1]
        prediction2 = pickle_model.predict_proba([x])[0][1]

        return Response({'XGB probability': prediction1, "Logistic regression": prediction2}, status=200)
