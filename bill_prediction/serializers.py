from bill.models import Bill
from rest_framework import serializers


class BillSerializer(serializers.ModelSerializer):
    class Meta:
        model = Bill
        fields = ['chamber', 'bill_id', 'introduced_at', 'status', 'policy_area', 'subjects',
                  'official_title', 'bill_type', 'amendment_bill', 'sponsors', 'co_sponsors']
