labels_name = [
    'Age',
    'Gender',
    'ResidencePlace',
    'Date',
    'EndDate',
    'Location',
    'Spot',
    'SocialRelation'
]

label2id, id2label = {}, {}


def convert_id_label(label_list):
    global label2id, id2label
    label2id = {'O': 0}
    idx = 1
    for key in label_list:
        label2id['B-{}'.format(key)] = idx
        idx += 1
    for key in label_list:
        label2id['I-{}'.format(key)] = idx
        idx += 1
    # for key in label_list:
    #     label2id['S-{}'.format(key)] = idx
    #     idx += 1

    for key, value in label2id.items():
        id2label[value] = key

convert_id_label(labels_name)
num_labels = len(label2id)
