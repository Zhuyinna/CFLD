## for copy
ps aux|grep wandb|grep -v grep | awk '{print $2}'|xargs kill -9

## 实验结果

| Method          | FID↓  | LPIPS↓ | SSIM↑ | PSNR↑ |
|-----------------|-------|--------|-------|-------|
| PIDM† [1] CVPR 23’ | 6.812 | 0.2006 | 0.6621| 15.630|
| PIDM‡ [1] CVPR 23’ | 6.440 | 0.1686 | 0.7109| 17.399|
| PoCoLD∗ [9] ICCV 23’| 8.067 | 0.1642 | 0.7310|       |
| CFLD (Ours)       | 6.804 | 0.1519 | 0.7378| 18.235|
| 1. decoder->CLIP-base | 30.810 | 0.5018 | 0.4906 | 8.423 |


## 改进思路
- [x] A. src_img通过clipvision提取768特征，替换原来的hidden_states【ref:PCDMs】
- [ ] B. source通过SD提取四个特征，再经过basictransformer分别得到up additional，可选择用于selfattention或者crossattention最后一个特征，8个transformer得到另一个encoder hidden states【giveup】
- [x] C. 用原来的swin transformer提取出的特征，在每个up-block后，加入Zero Cross-Attention Block【ref:stableVITON】
- [x] D. 改进损失函数，关键：查看stableVITON中的mask是什么
- [ ] test修改
- [ ] 模型权重的保存，看看是否要修改


## 训练/测试设置
### 方案A
1. sample设置
    - 4卡
    - bs=8
    - 2h 
### 方案C


## (TBD)
additional_residuals
key: block_11_cross_attn_q, value: torch.Size([8, 4096, 320])
key: block_10_cross_attn_q, value: torch.Size([8, 4096, 320])
key: block_9_cross_attn_q, value: torch.Size([8, 4096, 320])
key: block_8_cross_attn_q, value: torch.Size([8, 1024, 640])
key: block_7_cross_attn_q, value: torch.Size([8, 1024, 640])
key: block_6_cross_attn_q, value: torch.Size([8, 1024, 640])
key: block_5_cross_attn_q, value: torch.Size([8, 256, 1280])
key: block_4_cross_attn_q, value: torch.Size([8, 256, 1280])
key: block_3_cross_attn_q, value: torch.Size([8, 256, 1280])


    

