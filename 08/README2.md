### 项目来源

https://github.com/CLUEbenchmark/CLUENER2020/tree/master/pytorch_version

https://github.com/lonePatient/BERT-NER-Pytorch

后者的代码完成度比前者高

### 预训练模型

bert-base-chinese: Google的中文bert预训练模型 https://huggingface.co/transformers/v2.9.1/pretrained_models.html

reoberta_wwm_ext: 哈工大讯飞联合实验室预训练模型 https://github.com/ymcui/Chinese-BERT-wwm

### 对比

用span模型采用不同的预训练集对比

bert-base-chinese:

acc = 0.806829268292683
f1 = 0.8072230356271352
loss = 0.05454795124046287
recall = 0.8076171875

reoberta_wwm_ext:

acc = 0.8254764292878636
f1 = 0.8144482929242949
loss = 0.04605594078739586
recall = 0.8037109375

