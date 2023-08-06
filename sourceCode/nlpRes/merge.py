import json

bertRess = open("entities.txt", 'r', encoding='utf-8').readlines()
foolRess = open("foolNer.txt", 'r', encoding='utf-8').readlines()

with open(r'mergeRes.txt', 'a+', encoding='utf-8') as test:
    test.truncate(0)

for i in range(len(bertRess)):
    bertRes = json.loads(bertRess[i])
    foolRes = json.loads(foolRess[i])

    bertRes['entities'] += foolRes['entities']

    lenRes = len(bertRes['entities'])
    for j in range(lenRes - 1):
        for k in range(lenRes - 1):
            if bertRes['entities'][k][0] > bertRes['entities'][k + 1][0]:
                bertRes['entities'][k], bertRes['entities'][k + 1] = bertRes['entities'][k + 1], bertRes['entities'][k]

    article = json.dumps(bertRes, ensure_ascii=False)

    # 处理ner结果 （格式与entities.txt一致） 存入foolner.txt
    with open("mergeRes.txt", 'a', encoding='utf-8') as fp:
        fp.write(article)
        fp.write('\n')
