import joblib
from rest_framework import viewsets, status
from rest_framework.response import Response
from rest_framework.decorators import action
from bill.models import *
from .constants import *
from .helper.data_loader import FeatureExtractor
from .serializers import *
import os


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

        feature_extractor = FeatureExtractor()
        house_model_names = os.listdir(HOUSE_MODEL_PATH)
        senate_model_names = os.listdir(SENATE_MODEL_PATH)
        house_model_list = []
        senate_model_list = []

        for model_name in house_model_names:
            house_model_list.append(joblib.load(HOUSE_MODEL_PATH + model_name))
        for model_name in senate_model_names:
            senate_model_list.append(joblib.load(SENATE_MODEL_PATH + model_name))

        house_probs, senate_probs = [], []
        x_house = feature_extractor.get_features(bill, chamber='house', get_x=True)
        x_senate = feature_extractor.get_features(bill, chamber='senate', get_x=True)
        for model in house_model_list:
            house_probs.append(model.predict_proba([x_house])[0][1])
        house_probability = sum(house_probs) / len(house_probs)
        for model in senate_model_list:
            senate_probs.append(model.predict_proba([x_senate])[0][1])
        senate_probability = sum(senate_probs) / len(senate_probs)

        return Response({'house': house_probability, 'senate': senate_probability}, status=200)
