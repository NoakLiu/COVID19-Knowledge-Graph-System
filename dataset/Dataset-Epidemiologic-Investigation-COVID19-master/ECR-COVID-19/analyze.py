import json

instance = 0
num_entitiy = 0
with open('train.txt', 'r', encoding='utf-8') as f:
    for line in f.readlines():
        json_line = json.loads(line)
        instance += 1
        entity_list = json_line['entities']
        for entity in entity_list:
            if entity[-1] in ['Age','Gender','ResidencePlace','Date','EndDate','Location','Spot','SocialRelation']:
                num_entitiy += 1

print(instance, num_entitiy)

instance = 0
num_entitiy = 0
with open('valid.txt', 'r', encoding='utf-8') as f:
    for line in f.readlines():
        json_line = json.loads(line)
        instance += 1
        entity_list = json_line['entities']
        for entity in entity_list:
            if entity[-1] in ['Age', 'Gender', 'ResidencePlace', 'Date', 'EndDate', 'Location', 'Spot',
                              'SocialRelation']:
                num_entitiy += 1
print(instance, num_entitiy)

instance = 0
num_entitiy = 0
with open('test.txt', 'r', encoding='utf-8') as f:
    for line in f.readlines():
        json_line = json.loads(line)
        instance += 1
        entity_list = json_line['entities']
        for entity in entity_list:
            if entity[-1] in ['Age', 'Gender', 'ResidencePlace', 'Date', 'EndDate', 'Location', 'Spot',
                              'SocialRelation']:
                num_entitiy += 1
print(instance, num_entitiy)
