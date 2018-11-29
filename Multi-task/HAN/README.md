#### 代码

```
attention_layer.py   ------- 注意力机制实现
han_model.py         ------- 分层注意力机制 + 多任务学习实现
data_factory.py      ------- 数据工厂类
utils.py             ------- 工具类
prepare.py           ------- 数据预处理
train.py             ------- 训练
predict.py           ------- 预测
evaluate.py          ------- 评估
config.py            ------- 配置文件
main.py              ------- 主程序
```

#### 调试运行

调参和运行时，只需要 修改 config.py 里的值，然后运行 main.py 即可完成训练、预测、评估 整个流程。