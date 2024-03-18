## for copy
Lwd5gZOYP3p/


## 初次训练
单卡:   bs=4:   7h
        bs=8:   4.5h
        bs=12:  cuda out memory

## 改进
- [ ] target通过clipvision提取768特征，用于timeemd和encoder hidden states
- [ ] source通过SD提取四个特征，再经过basictransformer分别得到up additional，可选择用于selfattention或者crossattention
        最后一个特征，8个transformer得到另一个encoder hidden states
        

