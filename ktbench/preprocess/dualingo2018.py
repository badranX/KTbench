# -----------------------------------------------------------------------------
# Description: This file was adapted from dualingo 2018 task script
# -----------------------------------------------------------------------------

def load_data(filename):
    """
    This method loads and returns the data in filename. If the data is labelled training data, it returns labels too.

    Parameters:
        filename: the location of the training or test data you want to load.

    Returns:
        data: a list of InstanceData objects from that data type and track.
        labels (optional): if you specified training data, a dict of instance_id:label pairs.
    """

    # 'data' stores a list of 'InstanceData's as values.
    data = []

    # If this is training data, then 'labels' is a dict that contains instance_ids as keys and labels as values.
    training = False
    if filename.find('train') != -1:
        training = True

    if training:
        labels = dict()

    num_exercises = 0
    print('Loading instances...')
    instance_properties = dict()

    with open(filename, 'rt') as f:
        for line in f:
            line = line.strip()

            # If there's nothing in the line, then we're done with the exercise. Print if needed, otherwise continue
            if len(line) == 0:
                num_exercises += 1
                if num_exercises % 100000 == 0:
                    print('Loaded ' + str(len(data)) + ' instances across ' + str(num_exercises) + ' exercises...')
                instance_properties = dict()

            # If the line starts with #, then we're beginning a new exercise
            elif line[0] == '#':
                if 'prompt' in line:
                    instance_properties['prompt'] = line.split(':')[1]
                else:
                    list_of_exercise_parameters = line[2:].split()
                    for exercise_parameter in list_of_exercise_parameters:
                        [key, value] = exercise_parameter.split(':')
                        if key == 'countries':
                            value = value.split('|')
                        elif key == 'days':
                            value = float(value)
                        elif key == 'time':
                            if value == 'null':
                                value = None
                            else:
                                assert '.' not in value
                                value = int(value)
                        instance_properties[key] = value

            # Otherwise we're parsing a new Instance for the current exercise
            else:
                line = line.split()
                if training:
                    assert len(line) == 7
                else:
                    assert len(line) == 6
                assert len(line[0]) == 12

                instance_properties['instance_id'] = line[0]

                instance_properties['token'] = line[1]
                instance_properties['part_of_speech'] = line[2]

                instance_properties['morphological_features'] = dict()
                for l in line[3].split('|'):
                    [key, value] = l.split('=')
                    if key == 'Person':
                        value = int(value)
                    instance_properties['morphological_features'][key] = value

                instance_properties['dependency_label'] = line[4]
                instance_properties['dependency_edge_head'] = int(line[5])
                if training:
                    label = float(line[6])
                    labels[instance_properties['instance_id']] = label
                data.append(InstanceData(instance_properties=instance_properties))

        print('Done loading ' + str(len(data)) + ' instances across ' + str(num_exercises) +
              ' exercises.\n')

    if training:
        return data, labels
    else:
        return data


class InstanceData(object):
    """
    A bare-bones class to store the included properties of each instance. This is meant to act as easy access to the
    data, and provides a launching point for deriving your own features from the data.
    """
    def __init__(self, instance_properties):

        # Parameters specific to this instance
        self.instance_id = instance_properties['instance_id']
        self.token = instance_properties['token']
        self.part_of_speech = instance_properties['part_of_speech']
        self.morphological_features = instance_properties['morphological_features']
        self.dependency_label = instance_properties['dependency_label']
        self.dependency_edge_head = instance_properties['dependency_edge_head']

        # Derived parameters specific to this instance
        self.exercise_index = int(self.instance_id[8:10])
        self.token_index = int(self.instance_id[10:12])

        # Derived parameters specific to this exercise
        self.exercise_id = self.instance_id[:10]

        # Parameters shared across the whole session
        self.user = instance_properties['user']
        self.countries = instance_properties['countries']
        self.days = instance_properties['days']
        self.client = instance_properties['client']
        self.session = instance_properties['session']
        self.format = instance_properties['format']
        self.time = instance_properties['time']
        self.prompt = instance_properties.get('prompt', None)

        # Derived parameters shared across the whole session
        self.session_id = self.instance_id[:8]

import pandas as pd
import argparse
from pathlib import Path
import yamld

OUTPUTFOLDER = "output"

def process(training_file, outputfolder="middata", encoding="latin-1"):
    training_data, training_labels = load_data(training_file)

    def data2dict(datalist):
        return {k: [dat.__dict__[k] for dat in datalist] 
                for k,_  in datalist[0].__dict__.items()} 

    def mapper(x):
        id = x.instance_id
        label = training_labels[id]
        d = x.__dict__
        d['label'] = label
        return d
        
    training_data = map(mapper, training_data)
    # df_exer 相关处理

    # 处理列名
    df = pd.DataFrame(training_data)
    df['tmp_index'] = range(len(df))
    df['order_id'] = df[['time', 'days', 'user', 'tmp_index']].values.tolist()
    df['skill_id'] = df['token'].str.lower()

    # 修改列名
    df = df.rename(
        columns={'user': 'stu_id', 'exercise_id': 'exer_id', 'label': 'label',
                 'skill_id': 'kc_seq', 'order_id': 'order_id'})

    df = df.dropna(subset=["time", "days", "stu_id","exer_id", "kc_seq", "label"])
    # df_inter 的相关处理
    df_inter = df[['stu_id','exer_id','label', 'order_id']].copy()
    df_inter.drop_duplicates(['stu_id', 'exer_id', 'label'], keep='first', inplace=True)
    df_inter.sort_values('stu_id', inplace=True)

    # kc_seq
    print('start generating kc sequences...')
    df_exer = df[['exer_id', 'kc_seq']].copy()
    df_exer.drop_duplicates(inplace=True)
    df_exer = df_exer.groupby('exer_id')['kc_seq'].agg(list).reset_index()

    # # Save MidData

    feature2type = {
    'stu_id': 'token', 'exer_id': 'token', 'label': 'float', 'start_timestamp': 'float', 'cost_time': 'float',
                        'order_id': 'token',
                         'kc_seq': 'token_seq', 'assignment_id': 'token_seq'
    }
    df_inter.attrs['feature2type'] = feature2type
    df_exer.attrs['feature2type'] = feature2type
    Path(outputfolder).mkdir(parents=True, exist_ok=True)
    yamld.write_dataframe(f"{outputfolder}/inter.yaml", df_inter)
    yamld.write_dataframe(f"{outputfolder}/exer.yaml", df_exer)
    pd.set_option("mode.chained_assignment", "warn")  # igore warning


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='process middata')
    parser.add_argument('training_file', type=str, default='middata', help='The target directory (default: ./middata)')
    args = parser.parse_args()
    process(training_file=args.training_file)
