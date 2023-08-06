@echo off
chcp 65001>nul
echo 信息抽取中...
cd ./sourceCode/nlpProcess/bertBased/model
python cdc_ner.py --policy Bert_CRF
cd ../../foolNer/model
python cdc_ner.py --policy Bert_CRF
cd ../../../nlpRes
python merge.py
cd ../nlpToKG
echo 图谱生成中...
python generate.py
echo 全图计算中...
python GraphAlg.py
cd ../../neovis
start KG.html
