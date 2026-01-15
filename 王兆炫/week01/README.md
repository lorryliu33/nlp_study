本次作业采用的模型都是sklearn的经典模型,KNN和LR

使用了jieba进行中文分词

使用train_test_split验证了模型的好坏

```bash
--- 正在训练模型 1: KNN ---
KNN 模型在测试集上的准确率: 0.7103

--- 正在训练模型 2: 逻辑回归 ---
逻辑回归模型在测试集上的准确率: 0.8963
```
以及展示了一个判断分类例子

```bash
测试输入: '帮我打开客厅的灯'
KNN 预测类别: HomeAppliance-Control
逻辑回归 预测类别: HomeAppliance-Control
