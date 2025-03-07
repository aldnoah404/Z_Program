import numpy as np
import os
import cv2
import matplotlib.pyplot as plt

def compute_fft_energy(image):
    # 对图像进行二维快速傅里叶变换（FFT），得到复数形式的频域表示。
    f = np.fft.fft2(image)
    # 将零频率分量移动到频域中心，使得低频成分位于中心，高频成分位于边缘。
    f_shift = np.fft.fftshift(f)
    # 计算频域复数的幅度，得到频域幅度谱。
    magnitude = np.abs(f_shift)
    
    # 计算频域能量并归一化，使其值在 [0, 1] 之间。归一化后的能量表示图像中所有频率成分的总强度相对于图像像素总数的比例。
    total_energy = np.sum(magnitude**2) / (magnitude.size)

    # magnitude
    # --类型：二维 NumPy 数组，形状与输入图像相同。
    # --含义：频域幅度谱（Magnitude Spectrum）
    # total_energy
    # --类型：标量，表示频域能量。
    # --含义：图像在频域中的总能量，等于所有频域幅度的平方之和除以频域尺寸的平方。
    return magnitude, total_energy

def find_fundamental_frequency(magnitude):
    h, w = magnitude.shape
    cy, cx = h // 2, w // 2  # 频域中心
    
    # 只考虑上半部分（或右半部分）以避免重复计算
    y, x = np.ogrid[-cy:h-cy, -cx:w-cx]
    radius = np.sqrt(x**2 + y**2).flatten()
    magnitudes = magnitude.flatten()
    
    # 排除中心点附近的低频区域（如前15个像素）
    valid_indices = (radius > 15) & (magnitudes > 0)
    valid_radius = radius[valid_indices]
    valid_magnitudes = magnitudes[valid_indices]
    
    if len(valid_magnitudes) == 0:
        raise ValueError("No valid frequencies found")
    
    # 找到能量最大的频率
    max_index = np.argmax(valid_magnitudes)
    fundamental_frequency = valid_radius[max_index]
    
    # 确保返回的频率在合理的范围内
    if fundamental_frequency > min(h, w) // 2:
        raise ValueError("Fundamental frequency is out of reasonable range")
    
    # 在焊缝图像中，能量最大的频率对应的就是结构光的频率。因此，这里返回的就是结构光的基频。

    # fundamental_frequency
    # --类型：标量，表示频率。
    # --含义：能量最大的频率对应的频率值。
    return fundamental_frequency

def stripe_energy_ratio(magnitude, f0, bandwidth=2):
    if f0 <= 0 or bandwidth <= 0:
        raise ValueError("f0 and bandwidth must be positive numbers")
    
    h, w = magnitude.shape
    cy, cx = h // 2, w // 2  # 频域中心
    
    # 创建环形掩膜（提取以f0为中心的频带）
    y, x = np.ogrid[-cy:h-cy, -cx:w-cx]
    radius = np.sqrt(x**2 + y**2)
    mask = (radius >= (f0 - bandwidth)) & (radius <= (f0 + bandwidth))
    
    # 计算条纹能量占比
    if np.sum(mask) == 0:
        return 0  # 防止除零错误
    stripe_energy = np.sum(magnitude[mask]**2)
    total_energy = np.sum(magnitude**2)  # 使用未归一化的总能量
    energy_ratio = stripe_energy / total_energy

    # energy_ratio
    # --类型：标量，表示条纹能量占比。
    # --含义：条纹能量占比，等于条纹频域能量除以总频域能量。
    return energy_ratio
# 对指定数据集进行分类，指标为指定的阈值。并返回分类结果以及一个字典，其中包含每张图片的能量比。
def classify_images(image_folder, classification_threshold=0.02, image_type="default"):
    # 检查文件夹是否存在
    if not os.path.exists(image_folder):
        raise FileNotFoundError(f"Image folder not found: {image_folder}")

    # 获取文件夹中的所有图片路径
    image_paths = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]

    # 存储能量比的结果
    energy_ratios_dict = {}

    # 遍历每张图片
    for image_path in image_paths:
        if not cv2.haveImageReader(image_path):
            print(f"Image file not found or unsupported format: {image_path}")
            continue
        
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"Failed to read image: {image_path}")
            continue
        
        magnitude, _ = compute_fft_energy(image)
        
        try:
            fundamental_frequency = find_fundamental_frequency(magnitude)
            energy_ratio = stripe_energy_ratio(magnitude, fundamental_frequency)
            img_name = os.path.basename(image_path)
            energy_ratios_dict[img_name] = energy_ratio

        except ValueError as e:
            print(f"Error processing image {image_path}: {e}")

    # 根据分类阈值将图片分为两类
    class_1_indices = [i for i, (img_name, ratio) in enumerate(energy_ratios_dict.items()) if ratio >= classification_threshold]
    class_2_indices = [i for i, (img_name, ratio) in enumerate(energy_ratios_dict.items()) if ratio < classification_threshold]

    # 打印分类结果
    print(f"Class 1 (能量比 >= {classification_threshold}): {len(class_1_indices)} 张图片")
    print(f"Class 2 (能量比 < {classification_threshold}): {len(class_2_indices)} 张图片")

    # 创建保存分类图片的文件夹
    class_1_folder = os.path.join(image_folder, 'classified_images', f'class_1_{image_type}')
    class_2_folder = os.path.join(image_folder, 'classified_images', f'class_2_{image_type}')

    os.makedirs(class_1_folder, exist_ok=True)
    os.makedirs(class_2_folder, exist_ok=True)

    # 保存分类后的图片
    def save_images(indices, folder):
        for idx in indices:
            img_name = list(energy_ratios_dict.keys())[idx]
            img_path = os.path.join(image_folder, img_name)
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                print(f"Failed to read image: {img_path}")
                continue
            save_path = os.path.join(folder, img_name)
            cv2.imwrite(save_path, image)
            print(f"Saved image: {save_path}")

    # 保存 Class 1 的图片
    save_images(class_1_indices, class_1_folder)

    # 保存 Class 2 的图片
    save_images(class_2_indices, class_2_folder)

    # 保存能量比字典到文件
    energy_ratios_file = os.path.join(image_folder, f'energy_ratios_{image_type}.json')
    with open(energy_ratios_file, 'w') as f:
        import json
        json.dump(energy_ratios_dict, f)
    print(f"Saved energy ratios to: {energy_ratios_file}")
