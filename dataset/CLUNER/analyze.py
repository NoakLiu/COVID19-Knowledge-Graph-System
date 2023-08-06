import json

instance = 0
num_entitiy = 0
with open('train.json', 'r', encoding='utf-8') as f:
    for line in f.readlines():
        json_line = json.loads(line)
        instance += 1
        entity_list = json_line['label']
        for entity in entity_list.keys():
            if entity in ['name','position','company','government']:
                num_entitiy += 1

print(instance, num_entitiy)

instance = 0
num_entitiy = 0
with open('dev_origin.json', 'r', encoding='utf-8') as f:
    for line in f.readlines():
        json_line = json.loads(line)
        instance += 1
        entity_list = json_line['label']
        for entity in entity_list.keys():
            if entity in ['name','position','company','government']:
                num_entitiy += 1
print(instance, num_entitiy)

# instance = 0
# num_entitiy = 0
# with open('test_origin.json', 'r', encoding='utf-8') as f:
#     for line in f.readlines():
#         json_line = json.loads(line)
#         instance += 1
#         entity_list = json_line['label']
#         for entity in entity_list:
#             if entity[-1] in ['Age', 'Gender', 'ResidencePlace', 'Date', 'EndDate', 'Location', 'Spot',
#                               'SocialRelation']:
#                 num_entitiy += 1
# print(instance, num_entitiy)
