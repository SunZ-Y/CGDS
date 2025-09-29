import os
import numpy as np
import scipy.io as sio
import pickle
import gzip
from pathlib import Path
import json


def restore_mat_dataset(compressed_root, output_root, verify_integrity=True):
    """
    从压缩文件恢复原始.mat数据集，包括完整的文件名和目录结构

    参数:
    - compressed_root: 压缩文件目录
    - output_root: 恢复输出的目录
    - verify_integrity: 是否验证文件完整性
    """

    compressed_path = Path(compressed_root)
    output_path = Path(output_root)
    output_path.mkdir(parents=True, exist_ok=True)

    # 加载压缩信息
    info_file = compressed_path / "compression_info.pkl"
    if not info_file.exists():
        info_file = compressed_path / "compression_info.json"
        with open(info_file, 'r', encoding='utf-8') as f:
            compression_info = json.load(f)
    else:
        with open(info_file, 'rb') as f:
            compression_info = pickle.load(f)

    restored_files = []
    error_files = []

    print("开始恢复数据集...")

    # 遍历所有文件映射信息
    for file_info in compression_info['file_mapping']:
        try:
            # 创建输出目录
            output_dir = output_path / file_info['split'] / file_info['class']
            output_dir.mkdir(parents=True, exist_ok=True)

            # 输出文件路径（使用原始文件名）
            output_file = output_dir / file_info['original_filename']

            print(f"恢复: {output_file}")

            # 加载压缩数据
            compressed_file = compressed_path / file_info['compressed_filename']
            with gzip.open(compressed_file, 'rb') as f:
                compressed_data = pickle.load(f)

            # 反量化数据
            restored_data = {}
            variable_info = file_info['variable_info']

            for key, value in compressed_data.items():
                if key in variable_info:
                    var_info = variable_info[key]
                    if 'quantization_info' in var_info:
                        restored_data[key] = dequantize_data(value, var_info['quantization_info'])
                    else:
                        restored_data[key] = value
                else:
                    restored_data[key] = value

            # 保存为.mat文件
            sio.savemat(output_file, restored_data)

            # 验证文件完整性（可选）
            if verify_integrity:
                try:
                    # 检查文件是否能正常加载
                    test_data = sio.loadmat(output_file)
                    for expected_key in ['heatmaps', 'heatmaps5', 'add']:
                        if expected_key in variable_info and expected_key not in test_data:
                            print(f"警告: 文件 {output_file} 中缺少变量 {expected_key}")

                    restored_files.append({
                        'file': str(output_file),
                        'status': 'success',
                        'verified': True
                    })

                except Exception as e:
                    print(f"验证文件 {output_file} 时出错: {e}")
                    restored_files.append({
                        'file': str(output_file),
                        'status': 'warning',
                        'error': str(e)
                    })
            else:
                restored_files.append({
                    'file': str(output_file),
                    'status': 'success',
                    'verified': False
                })

        except Exception as e:
            print(f"恢复文件 {file_info['original_filename']} 时出错: {e}")
            error_files.append({
                'file': file_info['original_filename'],
                'error': str(e)
            })

    # 生成恢复报告
    generate_restoration_report(restored_files, error_files, compression_info)

    return restored_files, error_files


def dequantize_data(quantized_data, quant_info):
    """根据量化信息反量化数据"""
    method = quant_info.get('method', 'none')

    if method == 'float16':
        # float16 直接转换为 float32
        return quantized_data.astype(np.float32)
    elif method == 'int16':
        # int16 根据原始类型转换
        original_dtype = quant_info.get('original_dtype', 'int32')
        if 'int64' in original_dtype:
            return quantized_data.astype(np.int64)
        else:
            return quantized_data.astype(np.int32)
    elif method == 'float_uint8':
        # 反量化8位浮点数
        data_min = quant_info['min']
        data_max = quant_info['max']
        normalized = quantized_data.astype(np.float32) / 255.0
        restored = normalized * (data_max - data_min) + data_min
        return restored.astype(np.float32)
    elif method == 'int8':
        # int8 根据原始类型转换
        original_dtype = quant_info.get('original_dtype', 'int32')
        if 'int64' in original_dtype:
            return quantized_data.astype(np.int64)
        else:
            return quantized_data.astype(np.int32)
    else:
        # 没有量化，直接返回
        return quantized_data


def generate_restoration_report(restored_files, error_files, compression_info):
    """生成恢复报告"""
    print(f"\n=== 恢复完成报告 ===")
    print(f"成功恢复文件: {len(restored_files)}")
    print(f"失败文件: {len(error_files)}")

    if error_files:
        print(f"\n失败文件列表:")
        for error in error_files:
            print(f"  - {error['file']}: {error['error']}")

    # 按类别统计
    class_stats = {}
    for file_info in compression_info['file_mapping']:
        class_name = file_info['class']
        if class_name not in class_stats:
            class_stats[class_name] = 0
        class_stats[class_name] += 1

    print(f"\n=== 原始数据集结构 ===")
    for class_name, count in class_stats.items():
        print(f"类别 {class_name}: {count} 个文件")

    # 保存详细报告
    report = {
        'restoration_time': np.datetime64('now').astype(str),
        'successful_files': len(restored_files),
        'failed_files': len(error_files),
        'failed_list': error_files,
        'original_structure': class_stats
    }

    output_path = Path(compression_info.get('output_root', 'CGDS_restored'))
    report_file = output_path / "restoration_report.json"

    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n详细报告已保存至: {report_file}")


def check_dataset_integrity(original_root, restored_root, sample_per_class=2):
    """
    检查恢复数据集的完整性

    参数:
    - original_root: 原始数据集目录
    - restored_root: 恢复的数据集目录
    - sample_per_class: 每个类别抽样检查的文件数量
    """

    original_path = Path(original_root)
    restored_path = Path(restored_root)

    print(f"\n=== 数据集完整性检查 ===")

    checked_files = 0
    errors_found = 0

    for split_dir in ['train', 'val']:
        split_original = original_path / split_dir
        split_restored = restored_path / split_dir

        if not split_original.exists() or not split_restored.exists():
            print(f"警告: {split_dir} 目录不存在")
            continue

        for class_dir in sorted(split_original.iterdir()):
            if class_dir.is_dir():
                class_name = class_dir.name
                mat_files = list(class_dir.glob('*.mat'))[:sample_per_class]

                for mat_file in mat_files:
                    restored_file = split_restored / class_name / mat_file.name

                    if restored_file.exists():
                        try:
                            # 加载原始和恢复的数据
                            original_data = sio.loadmat(mat_file)
                            restored_data = sio.loadmat(restored_file)

                            # 比较关键变量
                            for key in ['heatmaps', 'heatmaps5', 'add']:
                                if key in original_data and key in restored_data:
                                    orig = original_data[key]
                                    rest = restored_data[key]

                                    if orig.shape == rest.shape:
                                        # 计算相对误差
                                        if np.issubdtype(orig.dtype, np.floating):
                                            mse = np.mean((orig - rest) ** 2)
                                            max_error = np.max(np.abs(orig - rest))
                                            print(
                                                f"{mat_file.name} - {key}: MSE = {mse:.6f}, 最大误差 = {max_error:.6f}")
                                        else:
                                            # 对于整数数据，检查是否完全一致
                                            if np.array_equal(orig, rest):
                                                print(f"{mat_file.name} - {key}: 数据完全一致")
                                            else:
                                                diff_count = np.sum(orig != rest)
                                                print(f"{mat_file.name} - {key}: 警告! {diff_count} 个值不同")
                                                errors_found += 1
                                    else:
                                        print(
                                            f"{mat_file.name} - {key}: 形状不匹配! 原始: {orig.shape}, 恢复: {rest.shape}")
                                        errors_found += 1

                            checked_files += 1

                        except Exception as e:
                            print(f"检查文件 {mat_file.name} 时出错: {e}")
                            errors_found += 1
                    else:
                        print(f"文件不存在: {restored_file}")
                        errors_found += 1

    print(f"\n完整性检查完成:")
    print(f"检查文件数: {checked_files}")
    print(f"发现错误: {errors_found}")

    return errors_found == 0


# 使用示例
if __name__ == "__main__":
    compressed_folder = "CGDS_compressed"
    restored_folder = "CGDS_restored"

    # 恢复数据集
    restored_files, error_files = restore_mat_dataset(compressed_folder, restored_folder)
'''
    # 可选：检查完整性
    original_folder = "CGDS"
    if check_dataset_integrity(original_folder, restored_folder):
        print("数据集完整性检查通过!")
    else:
        print("警告: 数据集完整性检查发现一些问题!")
'''