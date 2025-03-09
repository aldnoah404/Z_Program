import torch
from model import U_NeXt_v1,U_NeXt_v3,U_NeXt_v4
from fvcore.nn import FlopCountAnalysis, parameter_count
import json
import os
import time

def model_info_test(model_name="U_NeXt_v1"):
    # 创建模型实例
    model = U_NeXt_v1(in_channels=1, out_channels=1)
    if model_name == "U_NeXt_v3":
        model = U_NeXt_v3(in_channels=1, out_channels=1)
    elif model_name == "U_NeXt_v4":
        model = U_NeXt_v4(in_channels=1, out_channels=1)

    
    # 计算并打印模型参数数量
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型参数数量: {total_params}")

    # 创建一个随机输入张量
    input_tensor = torch.randn(1, 1, 288, 288)  # 批量大小为1，输入通道数为1，高度和宽度为288

    # 计算FLOPS
    flops = FlopCountAnalysis(model, input_tensor)
    print(f"模型FLOPS: {flops.total()}")

    # 测量模型推理速度
    model.eval()
    with torch.no_grad():
        start_time = time.time()
        for _ in range(100):  # 运行100次前向传播
            output_tensor = model(input_tensor)
        end_time = time.time()
    inference_time = (end_time - start_time) / 100  # 平均每次前向传播时间
    print(f"模型推理时间: {inference_time:.6f} 秒")

    # 进行前向传播
    output_tensor = model(input_tensor)

    # 打印输出张量的形状
    print(f"输出张量形状: {output_tensor.shape}")
    
    # 检查文件是否存在以及是否包含相同的模型信息
    if os.path.exists("model_info.json") and os.path.getsize("model_info.json") > 0:
        try:
            with open("model_info.json", "r") as f:
                existing_info = json.load(f)
        except json.JSONDecodeError:
            existing_info = {}
    else:
        existing_info = {}
    
    # 更新现有信息或添加新信息
    if model_name in existing_info:
        if "construction_info" in existing_info[model_name]:
            model_info = existing_info[model_name]["construction_info"]
        else:
            model_info = {}
    else:
        existing_info[model_name] = {}
        model_info = {}

    # 将数据存储到字典中
    model_info.update({
        "total_params": total_params,
        "flops": flops.total(),
        "inference_time": round(inference_time, 6)
    })
    

    # 更新现有信息或添加新信息
    existing_info[model_name]["construction_info"] = model_info

    # 将字典写入文件
    with open("model_info.json", "w") as f:
        json.dump(existing_info, f, indent=4)


if __name__ == "__main__":
    model_info_test(model_name="U_NeXt_v4")