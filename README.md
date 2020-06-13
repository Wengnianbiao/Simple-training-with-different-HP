# Simple-training-with-different-HP
Simple training in simple datasets via different hyper_parameters and methods to look for changes in accuracy.

## 训练背景
> &emsp;&emsp;本文使用101_ObjectCategories数据集，通过调整不同Epoch大小、不同学习率策略、不同模型、不同梯度下降算法等去查看精度的变化。  
### 1、Epoch
<p align="center"> 
<img src="https://raw.githubusercontent.com/Wengnianbiao/Simple-training-with-different-HP/master/figure_epoch/data10_10.jpg">
</p>
> ***Epoch:10***  
> **Best test Acc: 0.87**
<p align="center"> 
<img src="https://raw.githubusercontent.com/Wengnianbiao/Simple-training-with-different-HP/master/figure_epoch/data10_20.jpg">
</p>
> ***Epoch:20***  
> **Best test Acc: 0.88**
<p align="center"> 
<img src="https://raw.githubusercontent.com/Wengnianbiao/Simple-training-with-different-HP/master/figure_epoch/data10_50.jpg">
</p>
> ***Epoch:50***  
> **Best test Acc: 0.93**  
<p align="center"> 
<img src="https://raw.githubusercontent.com/Wengnianbiao/Simple-training-with-different-HP/master/figure_epoch/data10_100.jpg">
</p>
> ***Epoch:100***  
> **Best test Acc: 0.93**  
<p align="center"> 
<img src="https://raw.githubusercontent.com/Wengnianbiao/Simple-training-with-different-HP/master/figure_epoch/data10_150.jpg">
</p>
> ***Epoch:150***  
> **Best test Acc: 0.93**  

> &emsp;&emsp;分别使用了10、20、50、100、150个epoch训练模型。可以从实验图像看出，刚开始随着epoch的不断增大，测试集的精度也随之增加，但当epoch等于50的时候，精度维持在了
