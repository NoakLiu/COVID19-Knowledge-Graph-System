import sys
type = sys.getfilesystemencoding()
import json
from py2neo import *
import io
import sys
from dataStructure.Person import Person
from dataStructure.Place import Place
from dataStructure.Locate import Locate
from dataStructure.Lineage import Lineage

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# 连接neo4j数据库，输入地址、用户名、密码
graph = Graph('bolt://localhost:7687', auth=("neo4j", "zichaol"))
nodeMatcher = NodeMatcher(graph)
relationMatcher = RelationshipMatcher(graph)
graph.run('match (n) detach delete n')
file = open("../nlpRes/mergeRes.txt", 'r', encoding='utf-8')

lines = file.readlines()
# print(len(lines))
lineageFile = open("baseSet/lineage.txt", 'r', encoding='utf-8')
lineageSet = [i.strip() for i in lineageFile.readlines()]

personset = set()
placeset = set()


def isNotValid(str, set):
    for char in str:
        if char in set:
            return False
    return True

# 建立所有节点
for m in range(0, len(lines)):
    line = lines[m].strip()
    j = json.loads(line)
    text = j['text']
    entities = j['entities']
    if len(entities) <= 1:
        continue
    for k in range(len(entities)):
        entity = entities[k]

        # 对姓名进行处理
        if entity[-2] == 'name' and entity[-1].isalpha():
            p: Person = Person(entity[-1])
            per_node = Node('Person', name=p.name)
            graph.merge(per_node, "Person", "name")
            personset.add(entity[-1])

        # 对地点进行处理
        if entity[-2] in ['Location', 'ResidencePlace', 'Spot', 'org', 'company', 'government']:
            q: Place = Place(entity[-1], entity[-2])
            pla_node = Node('Place', name=q.name, type=q.type)
            graph.merge(pla_node, "Place", "name")
            placeset.add(entity[-1])

print("人物节点数：", len(personset))
print("地点节点数：", len(placeset))

# 匹配“亲缘”关系
cnt = 0
for m in range(0, len(lines)):
    line = lines[m].strip()
    j = json.loads(line)
    text = j['text']
    entities = j['entities']
    if len(entities) <= 1:
        continue
    for k in range(1, len(entities) - 1):
        entity = entities[k]
        if entity[-2] == 'SocialRelation':
            # if isNotValid(entity[-1], lineageSet):
            #     continue
            if entities[k - 1][-2] == 'name' and entities[k + 1][-2] == 'name' and entities[k - 1][-1] != entities[k + 1][-1] and entities[k - 1][-1].isalpha() and \
                    entities[k + 1][-1].isalpha():
                # print(entity[-1] + '-' + entities[k-1][-1] + '-' + entities[k+1][-1])
                l: Lineage = Lineage(entities[k - 1][-1], entity[-1], entities[k + 1][-1])
                sourceNode = nodeMatcher.match("Person", name=l.sourceName).first()
                sinkNode = nodeMatcher.match("Person", name=l.sinkName).first()
                if relationMatcher.match((sourceNode, sinkNode), r_type='Lineage').first() is None:
                    lineage = Relationship(sourceNode, 'Lineage', sinkNode)
                    lineage['specific'] = l.specific
                    graph.create(lineage)
                    cnt += 1
            elif entities[k - 1][-2] == 'name' and entities[0][-1] != entities[k - 1][-1] and entities[k - 1][-1].isalpha():
                # print(entity[-1] + '-' + entities[0][-1] + '-' + entities[k-1][-1])
                l: Lineage = Lineage(entities[0][-1], entity[-1], entities[k - 1][-1])
                sourceNode = nodeMatcher.match("Person", name=l.sourceName).first()
                sinkNode = nodeMatcher.match("Person", name=l.sinkName).first()
                if relationMatcher.match((sourceNode, sinkNode), r_type='Lineage').first() is None:
                    lineage = Relationship(sourceNode, 'Lineage', sinkNode)
                    lineage['specific'] = l.specific
                    graph.create(lineage)
                    cnt += 1
            elif entities[k + 1][-2] == 'name' and entities[0][-1] != entities[k + 1][-1] and entities[k + 1][
                -1].isalpha():
                # print(entity[-1] + '-' + entities[0][-1] + '-' + entities[k+1][-1])
                l: Lineage = Lineage(entities[0][-1], entity[-1], entities[k + 1][-1])
                sourceNode = nodeMatcher.match("Person", name=l.sourceName).first()
                sinkNode = nodeMatcher.match("Person", name=l.sinkName).first()
                if relationMatcher.match((sourceNode, sinkNode), r_type='Lineage').first() is None:
                    lineage = Relationship(sourceNode, 'Lineage', sinkNode)
                    lineage['specific'] = l.specific
                    graph.create(lineage)
                    cnt += 1

print("亲缘关系数：", cnt)

# 匹配”位于“关系
def isDate(str):
    dateStr = ["年", "月", "日", "时", "中午", "晚上", "上午", "早晨", "凌晨", "下午", "半夜"]
    for i in dateStr:
        if i in str:
            return True
    return False


cnt = 0
for m in range(len(lines)):
    personset = set()
    placeset = set()
    line = lines[m].strip()
    j = json.loads(line)
    text = j['text']
    entities = j['entities']
    if len(entities) <= 1:
        continue
    for entity in entities:
        if entity[-2] == 'name' and entity[-1].isalpha():
            personset.add(entity[-1])

    time = ' set() '
    for k in range(1, len(entities)):
        entity = entities[k]
        if entity[-2] == 'Date' and isDate(entity[-1]):
            time = entity[-1]
            matchRange = range(max(0, k - 5), min(len(entities), k + 5))
            for i in matchRange:
                if entities[i][-2] in ['Location', 'ResidencePlace', 'Spot', 'org', 'company', 'government']:
                    placeset.add(entities[i][-1])
            # print(personset, time, placeset)
            if len(personset) > 0 and len(placeset) > 0:
                for per in personset:
                    for pla in placeset:
                        sourceNode = nodeMatcher.match("Person", name=per).first()
                        sinkNode = nodeMatcher.match("Place", name=pla).first()
                        if relationMatcher.match((sourceNode, sinkNode), r_type='Locate').first() is None:
                            locate = Relationship(sourceNode, 'Locate', sinkNode)
                            locate['time'] = time
                            graph.create(locate)
                            cnt += 1

            time = entity[-1]
            placeset = set()

print("位于关系数：", cnt)

graph.run("""MATCH (n) WHERE size((n)--())=0 DELETE n""")
