result='''
[2024-03-19 05:02:07 pose_transfer_test.py:213] FID: 30.810
[2024-03-19 05:02:07 pose_transfer_test.py:214] LPIPS: 0.5018
[2024-03-19 05:02:07 pose_transfer_test.py:215] SSIM: 0.2026
[2024-03-19 05:02:07 pose_transfer_test.py:216] SSIM_256: 0.4906
[2024-03-19 05:02:07 pose_transfer_test.py:217] PSNR: 8.423
'''

exp_name = "decoder->CLIP-base"

# 解析输入文本，提取出评估指标的值
lines = result.strip().split("\n")
metrics = {}
for line in lines:
    parts = line.split("]")[1].split(":")
    key = parts[0].strip()
    value = parts[1].strip()
    metrics[key] = value

# 格式化输出
formatted_output = "| {} | {:>2} | {:>2} | {:>2} | {:>2} |".format(exp_name, metrics["FID"], metrics["LPIPS"], metrics["SSIM_256"], metrics["PSNR"])
print(formatted_output)