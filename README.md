# COVID19-Knowledge-Graph-System
A research project about construction of Covid19 spatial-tempal knowledge graph, then analysis, tracking, visuzalization of the potential infection paths on the knowledge graph.



![AVESA Framework for Covid-19 Analysis and Prediction](https://github.com/NoakLiu/Covid19-Knowledge-Graph-System/blob/main/AVESA_framework.png)



The main contributions of this paper are as follows: 

• I design and implement an end-to-end automated COVID-19 epidemiological analysis framework AVESA , which solves the inefficiency of traditional manual epidemiological analysis. 

• I design and implement a knowledge extraction model based on the combination of horizontal fusion, vertical fusion and deep learning, which solves the problem of low performance of directly transfer models trained in other fields. 

• I design and implement a complete set of the automated construction process of the epidemiological knowledge graph of COVID-19 and generate an epidemiological knowledge graph with 550 nodes and 1385 edges based on existing case reports text. 

• I establish this complete reasoning rules with different dimensions based on the epidemiological knowledge graph and present visualized results. These results provide an essential basis for epidemiological survey workers to select the next epidemic control strategies. 

• I design a series of comparative experiments for applications, conduct experiments, compare results, and finally demonstrate the feasibility and advantages of the AVESA framework.

# 

## 1. Automated analysis

root: sourceCode/nlpProcess/bertBased(foolNer)/model

**main.py：** Model training

**eval.py：** Model evaluation

**cdc_ner.py：** NER inference

### 1.1. Training(main.py)

Trained models will be saved in sourceCode/nlpProcess/bertBased(fullNer)/res/[model_name]

**Settings**

```
--batch-size     12
--epochs         5
--lr             3e-5
--resume         False
```

**Training**

```
python main.py --policy [model_name，such as Bert_MLP]
```

### 2.2. Evaluation(eval.py)

Evaluation results will be saved in sourceCode/nlpProcess/bertBased(fullNer)/res/eval.log

**Single model evaluation**

```
python eval.py --policy [model_name，such as Bert_MLP]
```

**Ensemble model evaluation**

```
python eval.py --ensemble [split with ","，such as Bert_MLP,BERT_CRF]
```

### 2.3. NER process(cdc_ner.py)

Outputs will be saved in 在sourceCode/nlpRes

**Single model**

```
python cdc_ner.py --policy [model_name，such as Bert_MLP]
```

**Ensemble model**

```
python cdc_ner.py --ensemble [split with ","，suchc as Bert_MLP,BERT_CRF]
```

##### Horizontal Fusion

```
cd sourceCode/nlpRes
python merge.py
```

Final results will be saved in mergeRes.txt，the content is as following：

```
{"text": "...", "entities": [[4, 7, "name", "..."], [8, 11, "name", "..."], [14, 17, "Age", "..."], [21, 23, "position", "..."], [70, 84, "ResidencePlace", "..."]}
...
```

## 3. KG

root: sourceCode/nlpToKG

### 3.1. Open Neo4j graph database

### 3.2. Construction of KG

```
python generate.py
```

### 3.3. Knowledge Reasoning

Refer to files in sourceCode/nlpToKG

## 4. Usage

Refer to pipeline.bat
