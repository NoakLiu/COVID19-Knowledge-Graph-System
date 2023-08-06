import os
import json
from docx import Document

#  定义指定路径 , 或者自己输入
path = r'origin_data'
# path = input("请输入路径:")
# 定义接收元组
vv = []


def read(path):
    for i in os.listdir(path):
        fi_d = os.path.join(path, i)
        # print(fi_d)
        if os.path.isdir(fi_d):
            # 调用递归
            read(fi_d)
        else:
            if os.path.splitext(i)[1] == '.docx':
                doc = Document(os.path.join(fi_d))
                for p in doc.paragraphs:
                    para = p.text.strip()
                    if len(para) > 20:
                        data = {'doc_id': os.path.splitext(i)[0],
                                'text': '此人名叫' + os.path.splitext(i)[0] + '。' + para}
                        article = json.dumps(data, ensure_ascii=False)
                        with open("test.txt", 'a', encoding='utf-8') as fp:
                            fp.write(article)
                            fp.write('\n')
                # vv.append(os.path.join(fi_d))


if __name__ == '__main__':
    with open(r'test.txt', 'a+', encoding='utf-8') as test:
        test.truncate(0)
    read(path)
