from bill.models import Bill, Vote
from directories.congress.models import LegislatorTerms
from bill_prediction.constants import *
from datetime import datetime
from django.db.models import Count
import json
import numpy as np
import pandas as pd
import os


class DataLoader:
    @classmethod
    def __convert_vote_key(cls, vote_key):
        if vote_key in ['yea', 'aye']:
            return 1
        elif vote_key in ['nay', 'no']:
            return 0
        else:
            return np.nan

    @classmethod
    def __convert_bill_status(cls, status):
        if status in PASS_STATUS_LIST:
            return 1
        elif status in FAIL_STATUS_LIST:
            return 0
        else:
            return np.nan

    @classmethod
    def __convert_house_bill_status(cls, status):
        if status in HOUSE_PASS_STATUS_LIST:
            return 1
        elif status in HOUSE_FAIL_STATUS_LIST:
            return 0
        else:
            return np.nan

    @classmethod
    def __convert_senate_bill_status(cls, status):
        if status in SENATE_PASS_STATUS_LIST:
            return 1
        elif status in SENATE_FAIL_STATUS_LIST:
            return 0
        else:
            return np.nan

    @classmethod
    def __generate_legislator_policy_relation(cls, current_votes):
        votes_data = current_votes.values('legislator__bioguide', 'vote__bill__policy_area', 'vote_key')
        votes_df = pd.DataFrame(votes_data)
        votes_df.rename(columns={
            'legislator__bioguide': 'legislator',
            'vote__bill__policy_area': 'policy_area'}, inplace=True)

        # Converting vote key to 1 for positive and 0 for the negative and nan for the rest
        votes_df['num_key'] = votes_df['vote_key'].apply(cls.__convert_vote_key)
        votes_df.dropna(inplace=True)
        votes_df.reset_index(drop=True, inplace=True)

        # Converting policy_area text to lower
        votes_df['policy_area'] = votes_df['policy_area'].apply(lambda x: x.lower())

        pivot_table = votes_df.pivot_table(index='legislator',
                                           columns='policy_area',
                                           values='num_key',
                                           aggfunc={'num_key': np.mean},
                                           fill_value=0.5)

        pivot_table.to_csv(LEGISLATOR_POLICY_AREA_FILE_PATH)
        print(f"File saved at {LEGISLATOR_POLICY_AREA_FILE_PATH}")

    @classmethod
    def __generate_legislator_subject_relation(cls, current_votes):
        votes_data = current_votes.values('legislator__bioguide', 'vote__bill__subjects', 'vote_key')
        subjects_data = []

        for vote in votes_data:
            subjects = vote['vote__bill__subjects']
            legislator = vote['legislator__bioguide']
            vote_key = vote['vote_key']

            for subject in subjects:
                subjects_data.append({'subject': subject.lower(),
                                      'legislator': legislator,
                                      'vote_key': vote_key})
        del votes_data

        subjects_df = pd.DataFrame(subjects_data)
        del subjects_data

        subjects_df['num_key'] = subjects_df['vote_key'].apply(cls.__convert_vote_key)
        subjects_df.dropna(inplace=True)
        subjects_df.reset_index(drop=True, inplace=True)

        grouped_df = subjects_df.groupby(['legislator', 'subject']).mean()
        del subjects_df

        grouped_df.reset_index(inplace=True)
        pivot_table = grouped_df.pivot_table(index='legislator',
                                             columns='subject',
                                             values='num_key',
                                             fill_value=0.5)
        del grouped_df
        pivot_table.to_csv(LEGISLATOR_SUBJECTS_FILE_PATH)
        print(f"File saved at {LEGISLATOR_SUBJECTS_FILE_PATH}")

    @classmethod
    def __generate_house_policy_subjects_relation(cls):
        bills = Bill.objects.filter(status__in=HOUSE_PASS_STATUS_LIST+HOUSE_FAIL_STATUS_LIST)\
                            .values('status', 'policy_area', 'subjects')

        bills_df = pd.DataFrame(bills)
        bills_df.dropna(inplace=True)
        bills_df.reset_index(drop=True, inplace=True)

        bills_df['policy_area'] = bills_df['policy_area'].apply(lambda x: x.lower())
        bills_df['status'] = bills_df['status'].apply(cls.__convert_house_bill_status)

        policy_area_df = bills_df[['policy_area', 'status']].groupby(['policy_area']).mean()
        policy_area_df.columns = ['ratio']
        policy_area_df.to_csv(HOUSE_POLICY_AREA_FILE_PATH)
        print(f"File saved at {HOUSE_POLICY_AREA_FILE_PATH}")

        subject_list = []
        for subjects, status in bills_df[['subjects', 'status']].values:
            for subject in subjects:
                subject_list.append({'subject': subject.lower(), 'status': status})
        subject_df = pd.DataFrame(subject_list).groupby(['subject']).mean()
        subject_df.rename(columns={'status': 'ratio'}, inplace=True)
        subject_df.to_csv(HOUSE_SUBJECTS_FILE_PATH)
        print(f"File saved at {HOUSE_SUBJECTS_FILE_PATH}")

    @classmethod
    def __generate_senate_policy_subjects_relation(cls):
        bills = Bill.objects.filter(status__in=SENATE_PASS_STATUS_LIST+SENATE_FAIL_STATUS_LIST)\
                            .values('status', 'policy_area', 'subjects')

        bills_df = pd.DataFrame(bills)
        bills_df.dropna(inplace=True)
        bills_df.reset_index(drop=True, inplace=True)

        bills_df['policy_area'] = bills_df['policy_area'].apply(lambda x: x.lower())
        bills_df['status'] = bills_df['status'].apply(cls.__convert_senate_bill_status)

        policy_area_df = bills_df[['policy_area', 'status']].groupby(['policy_area']).mean()
        policy_area_df.columns = ['ratio']
        policy_area_df.to_csv(SENATE_POLICY_AREA_FILE_PATH)
        print(f"File saved at {SENATE_POLICY_AREA_FILE_PATH}")

        subject_list = []
        for subjects, status in bills_df[['subjects', 'status']].values:
            for subject in subjects:
                subject_list.append({'subject': subject.lower(), 'status': status})
        subject_df = pd.DataFrame(subject_list).groupby(['subject']).mean()
        subject_df.rename(columns={'status': 'ratio'}, inplace=True)
        subject_df.to_csv(SENATE_SUBJECTS_FILE_PATH)
        print(f"File saved at {SENATE_SUBJECTS_FILE_PATH}")

    @classmethod
    def generate_statistics_files(cls):
        os.makedirs('bill_prediction/outputs', exist_ok=True)

        # Setting up current government details
        print("...Generating current government details...")
        # Setting current government as the party with the highest number of members
        parties = LegislatorTerms.objects.filter(end_date__gt=datetime.now()).values('party'). \
            annotate(count=Count('party')).order_by()
        party_dict = {p['party']: p['count'] for p in parties}
        current_government = max(party_dict, key=lambda x: party_dict[x])

        # Set the current house members and senate members
        current_legs = LegislatorTerms.objects.filter(end_date__gt=datetime.now()). \
            values('legislator__bioguide', 'term_type')
        current_house_members = [leg['legislator__bioguide'] for leg in current_legs if leg['term_type'] == 'rep']
        current_senate_members = [leg['legislator__bioguide'] for leg in current_legs if leg['term_type'] == 'sen']

        current_details = {
            'government': current_government,
            'house members': current_house_members,
            'senate members': current_senate_members}
        json.dump(current_details, open(CURRENT_GOVERNMENT_DETAILS_FILE_PATH, 'w'))

        # Get all the previous votes of the current legislators
        current_votes = Vote.objects.filter(legislator__bioguide__in=current_house_members + current_senate_members)
        print("...Generating legislator-policy_area statistics...")
        cls.__generate_legislator_policy_relation(current_votes)
        print("...Generating legislator-subjects statistics...")
        cls.__generate_legislator_subject_relation(current_votes)
        print("...Generating House of Representatives policy area and subjects statistics...")
        cls.__generate_house_policy_subjects_relation()
        print("...Generating Senate policy area and subjects statistics...")
        cls.__generate_senate_policy_subjects_relation()


class FeatureExtractor:
    def __init__(self):
        try:
            government_details = json.load(open(CURRENT_GOVERNMENT_DETAILS_FILE_PATH))
            self.__current_government = government_details['government']
            self.__current_house_members = government_details['house members']
            self.__current_senate_members = government_details['senate members']
            self.__legis_policy_df = pd.read_csv(LEGISLATOR_POLICY_AREA_FILE_PATH, index_col='legislator')
            self.__legis_subject_df = pd.read_csv(LEGISLATOR_SUBJECTS_FILE_PATH, index_col='legislator')
            self.__house_policy_df = pd.read_csv(HOUSE_POLICY_AREA_FILE_PATH, index_col='policy_area')
            self.__house_subject_df = pd.read_csv(HOUSE_SUBJECTS_FILE_PATH, index_col='subject')
            self.__senate_policy_df = pd.read_csv(SENATE_POLICY_AREA_FILE_PATH, index_col='policy_area')
            self.__senate_subject_df = pd.read_csv(SENATE_SUBJECTS_FILE_PATH, index_col='subject')
            self.__gov_mapper = {
                '107': 'Republican',
                '108': 'Republican',
                '109': 'Republican',
                '110': 'Democrat',
                '111': 'Democrat',
                '112': 'Republican',
                '113': 'Republican',
                '114': 'Democrat',
                '115': 'Democrat',
                '116': 'Republican',
                '117': 'Democrat',
            }

        except FileNotFoundError:
            raise Exception("""At least one of the statistics file is missing. Run python manage.py train_model command without --use_cache argument""")

    def __legislator_policy_probability(self, term_type, policy, sponsors, co_sponsors):
        try:
            legis_policy_col = self.__legis_policy_df[policy.lower()]
        except (AttributeError, KeyError):
            return 0.5

        prob_dict = {}
        members = []
        if term_type == 'house':
            members = self.__current_house_members
        if term_type == 'senate':
            members = self.__current_senate_members

        for member in members:
            try:
                prob_dict[member] = legis_policy_col[member]
            except KeyError:
                prob_dict[member] = 0.5
        for leg in sponsors:
            prob_dict[leg.bioguide] = 1
        for leg in co_sponsors:
            prob_dict[leg.bioguide] = 1
        prob_values = sorted(list(prob_dict.values()), reverse=True)
        # return np.mean(prob_values[:len(prob_values)//2])
        return np.mean(prob_values)

    def __legislator_subject_probability(self, term_type, subjects, sponsors, co_sponsors):
        eligible_subjects = [subject.lower() for subject in subjects if subject.lower() in self.__legis_subject_df.columns]
        if len(eligible_subjects) > 0:
            mean_df = self.__legis_subject_df[eligible_subjects].mean(axis=1)
            members = []
            if term_type == 'house':
                members = self.__current_house_members
            if term_type == 'senate':
                members = self.__current_senate_members

            prob_dict = {}
            for member in members:
                try:
                    prob_dict[member] = mean_df[member]
                except KeyError:
                    prob_dict[member] = 0.5
            for leg in sponsors:
                prob_dict[leg.bioguide] = 1
            for leg in co_sponsors:
                prob_dict[leg.bioguide] = 1
            prob_values = sorted(list(prob_dict.values()), reverse=True)
            # return np.mean(prob_values[:len(prob_values) // 2])
            return np.mean(prob_values)
        return 0.5

    def get_features(self, bill: Bill, chamber, get_x=False, get_X_dict=False):
        # policy = bill.policy_area
        # subjects = bill.subjects
        status = bill.status
        bill_id = bill.bill_id
        sponsors = bill.sponsors.all()
        co_sponsors = bill.co_sponsors.all()

        # 1: Number of amendment, for bill it will be 0
        amendment_count = 0
        while bill.bill_type == 'amendment':
            amendment_count += 1
            bill = bill.amendment_bill
        policy = bill.policy_area
        subjects = bill.subjects

        if chamber == 'house':
            policy_df = self.__house_policy_df
            subject_df = self.__house_subject_df
            PASS_STATUS_LIST = HOUSE_PASS_STATUS_LIST
            FAIL_STATUS_LIST = HOUSE_FAIL_STATUS_LIST
        elif chamber == 'senate':
            policy_df = self.__senate_policy_df
            subject_df = self.__senate_subject_df
            PASS_STATUS_LIST = SENATE_PASS_STATUS_LIST
            FAIL_STATUS_LIST = SENATE_FAIL_STATUS_LIST

        # 2: Policy area probability in the chamber
        try:
            chamber_policy_prob = policy_df.loc[policy.lower(), 'ratio']
        except (KeyError, AttributeError):
            chamber_policy_prob = 0.5

        # 3: Subjects probability in the chamber
        subject_prob_list = []
        chamber_subject_prob = 0.5
        for subject in subjects:
            try:
                subject_prob_list.append(subject_df.loc[subject.lower(), 'ratio'])
            except KeyError:
                subject_prob_list.append(0.5)
        if len(subjects) > 0:
            chamber_subject_prob = np.mean(subject_prob_list)

        # 4: Policy area based on the legislators of the chamber
        # 5: Subjects based on the legislators of the chamber
        if chamber == 'house':
            legis_policy_prob = self.__legislator_policy_probability('house', policy, sponsors, co_sponsors)
            legis_subject_prob = self.__legislator_subject_probability('house', subjects, sponsors, co_sponsors)
        elif chamber == 'senate':
            legis_policy_prob = self.__legislator_policy_probability('senate', policy, sponsors, co_sponsors)
            legis_subject_prob = self.__legislator_subject_probability('senate', subjects, sponsors, co_sponsors)

        # 6: Minority of the sponsor
        current_government = self.__gov_mapper[bill.bill_id[-3:]]
        sponsor_minority = False
        for sponsor in sponsors:
            sp = LegislatorTerms.objects.filter(legislator=sponsor,
                                                start_date__lte=bill.introduced_at,
                                                end_date__gte=bill.introduced_at).first()
            if sp and sp.party != current_government:
                sponsor_minority = True

        # 7: Number of cosponsors count
        co_sponsor_count = co_sponsors.count()

        # 8: Whether the bill is originated from the same chamber
        if bill_id[0] == chamber[0]:
            origin_chamber = 1
        else:
            origin_chamber = 0

        # Label
        if status in PASS_STATUS_LIST or status == 'pass':
            label = 1
        elif status in FAIL_STATUS_LIST or status == 'fail':
            label = 0

        if get_x:
            return [chamber_policy_prob, chamber_subject_prob, legis_policy_prob, legis_subject_prob, sponsor_minority,
                    co_sponsor_count, amendment_count, origin_chamber]
        if get_X_dict:
            return {'independent policy': chamber_policy_prob,
                    'independent subject': chamber_subject_prob,
                    'legislator dependent policy': legis_policy_prob,
                    'legislator dependent subject': legis_subject_prob,
                    'sponsor minority': sponsor_minority,
                    'number of co sponsors': co_sponsor_count,
                    'bill amendment count': amendment_count,
                    'Bill origin': origin_chamber}

        return {'independent policy': chamber_policy_prob,
                'independent subject': chamber_subject_prob,
                'legislator dependent policy': legis_policy_prob,
                'legislator dependent subject': legis_subject_prob,
                'sponsor minority': sponsor_minority,
                'number of co sponsors': co_sponsor_count,
                'bill amendment count': amendment_count,
                'Bill origin': origin_chamber,
                'label': label}
