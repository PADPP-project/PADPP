import json
import copy
from base.dataset import EmotionalSupportDataset


class ESConv(EmotionalSupportDataset):

    def __init__(self, dataset_config, **kwargs):
        """
        constructor for class Cariglist Bargain dataset
        :param dataset_config: the configuration of the dataset
        :param kwargs: other keywords parameters
        """
        super().__init__(dataset_config, **kwargs)

    def read_data(self, data_path):
        """
        method that load the data from a file path
        :param data_path: the path to the dataset
        :return: a list of raw data
        """
        with open(data_path, 'r') as f:
            data = f.readlines()
            assert len(data) > 0
            return data

    def repurpose_dataset(self, data):
        """
        method that repurpose the dataset for the negotiation scenario
        :param data: the loaded raw data
        :return: a list of re-purposed data
        """
        new_data = []
        # for negotiation, there is no need for re-purposing the dataset
        # we employ a simple "processing" step here to keep the pipeline consistent
        for line in data:
            line = json.loads(line)
            new_data.append(line)
        return new_data

    def process_data(self, data):
        """
        method that process the dataset
        :param data: the loaded data
        :return: pre-processed data instances
        """
        all_instances = []
        for conv_id, line in enumerate(data):
            instances = self.construct_instances(conv_id, line)
            all_instances.extend(instances)
        return all_instances

    def construct_instances(self, conv_id, conv):
        """
        method that processes the data to obtain a list of instances
        :param conv_id: the id of the conversation
        :param conv: the conversation data
        :return: a list of instances.
        """
        instances = []
        task_background = {
            "emotion_type": conv['emotion_type'],
            "problem_type": conv['problem_type'],
            "situation": conv['situation'],
        }
        utts = []
        goals = ["None"]

        for utt in conv['dialog']:
            # user turn
            # in the bargain, the user is the seller
            # the system is the buyer
            if utt['speaker'] == 'usr':
                utts.append({'role': 'user', 'content': utt['text']})
            # the system turn
            elif utt['speaker'] == 'sys':
                goal = utt['strategy']
                self.goals.append(goal)
                # constructing an instance.
                instance = {
                    "conv_id": conv_id,
                    "response": utt['text'],
                    "goal": goal,
                    "pre_goals": copy.deepcopy(goals),
                    "dialogue_context": copy.deepcopy(utts),
                    "task_background": copy.deepcopy(task_background),
                }
                instances.append(instance)
                # update the dialogue context
                utts.append({'role': 'assistant', 'content': utt['text']})
                # update the goal path
                goals.append(goal)

        return instances
