import os
import re
import csv
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
from glob import glob
import tempfile
import shutil


def delete_specific_bands(input_folder, delete_ranges=None):
    """删除指定波段范围的文件"""
    if delete_ranges is None:
        delete_ranges = [
            (1, 7),
            (58, 76),
            (120, 128),
            (170, 213),
            (225, 242)
        ]

    pattern = re.compile(r'(?:band|b|_)(\d+)', re.IGNORECASE)

    deleted_count = 0
    error_files = []

    for filename in os.listdir(input_folder):
        file_path = os.path.join(input_folder, filename)
        if not os.path.isfile(file_path):
            continue

        match = pattern.search(filename)
        if not match:
            continue

        try:
            band_num = int(match.group(1))
        except ValueError:
            continue

        to_delete = any(start <= band_num <= end for start, end in delete_ranges)
        if to_delete:
            try:
                os.remove(file_path)
                deleted_count += 1
            except Exception as e:
                error_files.append((filename, str(e)))

    return len(error_files) == 0


def process_single_image(image_path, output_dir):
    """处理单张影像并转换为CSV（保留所有像元）"""
    try:
        filename = os.path.splitext(os.path.basename(image_path))[0]
        output_path = os.path.join(output_dir, f"{filename}.csv")

        # 尝试导入PIL库
        try:
            from PIL import Image
        except ImportError:
            print("错误: 请先安装PIL库 (pip install Pillow)")
            return False

        with Image.open(image_path) as img:
            if hasattr(img, 'n_frames') and img.n_frames > 1:
                img.seek(0)
            data = np.array(img)

        # 处理多通道图像（如RGB）
        if data.ndim == 3:
            if data.shape[2] >= 1:
                data = data[:, :, 0]  # 取第一个波段
            else:
                return False

        df = pd.DataFrame(data)
        df.to_csv(output_path, index=False, header=False)
        return True

    except Exception as e:
        print(f"处理影像文件时出错 {image_path}: {e}")
        return False


def batch_process_images(input_dir, output_dir):
    supported_extensions = ['.tif', '.tiff', '.img']
    image_files = []
    for ext in supported_extensions:
        image_files.extend(glob(os.path.join(input_dir, f'*{ext}')))
    image_files = sorted(set(image_files))

    if not image_files:
        raise ValueError(f"在目录 '{input_dir}' 中未找到任何支持的影像文件")

    success_count = 0
    for img_path in tqdm(image_files, desc="影像转换进度", unit="文件", ncols=100, colour='green', leave=False):
        if process_single_image(img_path, output_dir):
            success_count += 1

    return success_count > 0


def process_single_csv(csv_path, output_dir):
    try:
        df = pd.read_csv(csv_path, header=None)
        df = df.replace(0, "")
        filename = os.path.basename(csv_path)
        output_path = os.path.join(output_dir, filename)
        df.to_csv(output_path, index=False, header=False)
        return True
    except Exception:
        return False


def batch_process_csv(input_dir, output_dir):
    csv_files = glob(os.path.join(input_dir, "*.csv"))
    if not csv_files:
        raise ValueError(f"在目录 '{input_dir}' 中未找到任何CSV文件")

    success_count = 0
    for csv_path in tqdm(csv_files, desc="零值转换进度", unit="文件", ncols=100, colour='green', leave=False):
        if process_single_csv(csv_path, output_dir):
            success_count += 1

    return success_count > 0


def process_large_csv_files_with_progress(input_dir, output_dir, num_classes=5):
    if not os.path.exists(input_dir) or not os.path.isdir(input_dir):
        return False

    csv_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.csv')]
    if not csv_files:
        return False

    processed_count = 0
    for filename in tqdm(csv_files, desc="数据分类进度", unit="文件", ncols=100, colour='green', leave=False):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)

        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                rows = list(reader)

            if len(rows) < 2 or len(rows[0]) < 2:
                raise ValueError("数据不足")

            data_values = []
            for i in range(1, len(rows)):
                for j in range(1, len(rows[i])):
                    cell = rows[i][j]
                    try:
                        val = float(cell)
                        if not np.isnan(val):
                            data_values.append(val)
                    except (ValueError, TypeError):
                        pass

            if not data_values:
                raise ValueError("无有效数值")

            # 手动设置分类数量
            if num_classes > 1:
                quantiles = [(i + 1) / num_classes for i in range(num_classes - 1)]
                thresholds = np.quantile(data_values, quantiles)
                thresholds = sorted(thresholds)
            else:
                thresholds = []

            def classify(x, thresholds, num_classes):
                try:
                    x = float(x)
                    if num_classes == 1:
                        return '1'
                    for i, threshold in enumerate(thresholds):
                        if x <= threshold:
                            return str(i + 1)
                    return str(num_classes)
                except:
                    return str(x)

            new_rows = []
            for i, row in enumerate(rows):
                new_row = []
                for j, cell in enumerate(row):
                    if i == 0 or j == 0:
                        new_row.append(cell)
                    else:
                        new_row.append(classify(cell, thresholds, num_classes))
                new_rows.append(new_row)

            with open(output_path, 'w', encoding='utf-8', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(new_rows)

            processed_count += 1

        except Exception as e:
            print(f"处理文件 {filename} 时出错: {e}")

    return processed_count > 0


def merge_csv_cells_by_position(input_dir, output_dir, filename, adjust_sizes=True):
    """
    增强版CSV合并函数，支持不同尺寸的CSV文件合并
    adjust_sizes: 是否自动调整不同尺寸的CSV文件
    """
    if not filename.lower().endswith('.csv'):
        filename += '.csv'
    output_file = os.path.join(output_dir, filename)

    if not os.path.exists(input_dir) or not os.path.isdir(input_dir):
        return False

    csv_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.csv')]
    if not csv_files:
        return False

    merged_data = None
    target_rows = 0
    target_cols = 0
    target_headers = None
    target_first_col = None

    file_paths = [os.path.join(input_dir, f) for f in csv_files]

    valid_files = 0

    with tqdm(total=len(file_paths), desc="📊 读取并合并文件", unit="file", colour='green') as pbar:
        for idx, file_path in enumerate(file_paths):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    reader = csv.reader(f)
                    rows = list(reader)

                # 跳过空文件
                if not rows or not rows[0]:
                    tqdm.write(f"\n⚠️ 跳过文件 {os.path.basename(file_path)}：文件为空")
                    pbar.update(1)
                    continue

                current_rows = len(rows)
                current_cols = max(len(row) for row in rows) if rows else 0  # 防止某行列数不一致

                # 尺寸调整逻辑
                if target_rows == 0:
                    # 第一个文件设为基准
                    target_rows = current_rows
                    target_cols = current_cols
                    # 初始化合并容器
                    merged_data = [[[] for _ in range(target_cols)] for _ in range(target_rows)]
                    # 记录首行首列
                    target_headers = rows[0][:target_cols] if current_rows > 0 else []
                    target_first_col = [row[0] if len(row) > 0 else '' for row in rows[:target_rows]]
                    tqdm.write(f"📏 设定基准尺寸: {target_rows} x {target_cols}")

                # 如果需要调整尺寸，则调整当前文件以匹配基准
                if adjust_sizes:
                    # 调整行数
                    if current_rows > target_rows:
                        # 裁剪多余的行
                        rows = rows[:target_rows]
                    elif current_rows < target_rows:
                        # 填充缺失的行
                        for _ in range(target_rows - current_rows):
                            rows.append([''] * min(current_cols, target_cols))

                    # 调整列数 (遍历每一行)
                    for i in range(len(rows)):
                        row = rows[i]
                        if len(row) > target_cols:
                            rows[i] = row[:target_cols]  # 裁剪
                        elif len(row) < target_cols:
                            # 填充空字符串
                            rows[i].extend([''] * (target_cols - len(row)))

                    # 确保当前文件尺寸与基准一致
                    current_rows = len(rows)
                    current_cols = len(rows[0]) if rows else 0

                # 如果不需要调整尺寸，则以第一个文件的尺寸为准，跳过不符合尺寸的文件
                else:
                    if current_rows != target_rows or current_cols != target_cols:
                        tqdm.write(
                            f"\n⚠️ 跳过文件 {os.path.basename(file_path)}：尺寸不符 ({current_rows}x{current_cols})，应为 ({target_rows}x{target_cols})")
                        pbar.update(1)
                        continue

                # 此时 rows 的尺寸应该是 target_rows x target_cols
                # 开始合并数据
                for i in range(min(target_rows, len(rows))):
                    for j in range(min(target_cols, len(rows[i]))):
                        cell = rows[i][j] if j < len(rows[i]) else ''

                        # 首行首列：只保留第一个文件的值
                        if i == 0 or j == 0:
                            if idx == 0 or not merged_data[i][j]:  # 如果是第一个文件，或者该位置还是空的
                                merged_data[i][j] = cell
                            continue

                        try:
                            # 转为浮点再格式化，去除多余0
                            val = f"{float(cell):.6g}" if cell != '' else ''
                            if val:  # 如果不是空值
                                merged_data[i][j].append(val)
                        except (ValueError, TypeError):
                            if cell != '':
                                merged_data[i][j].append(str(cell))

                valid_files += 1

            except Exception as e:
                tqdm.write(f"\n❌ 读取/处理失败 [{os.path.basename(file_path)}]: {e}")

            pbar.update(1)

    if merged_data is None:
        return False

    # 构建最终结果
    final_rows = []
    for i in range(target_rows):
        new_row = []
        for j in range(target_cols):
            cell = merged_data[i][j]
            if i == 0 or j == 0:
                # 确保首行列是字符串
                new_row.append(str(cell) if cell is not None else '')
            else:
                # 合并列表中的数值
                if isinstance(cell, list):
                    new_row.append(" ".join(cell))
                else:
                    new_row.append(str(cell))
        final_rows.append(new_row)

    # 写入输出文件
    try:
        with open(output_file, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(final_rows)
        return True
    except Exception:
        return False


def get_delete_ranges_from_user():
    print("请选择删除波段范围的选项：")
    print("1. 使用默认删除范围：(1, 7), (58, 76), (120, 128), (170, 213), (225, 242)")
    print("2. 自定义删除范围")

    choice = input("请输入选择 (1 或 2): ").strip()

    if choice == "1":
        return [(1, 7), (58, 76), (120, 128), (170, 213), (225, 242)]
    elif choice == "2":
        delete_ranges = []
        print("请输入自定义的删除范围，格式为 '开始-结束'，每行一个范围，输入空行结束：")
        while True:
            line = input("输入范围 (如 1-5) 或空行结束: ").strip()
            if line == "":
                break
            try:
                start, end = map(int, line.split('-'))
                delete_ranges.append((start, end))
            except ValueError:
                print("输入格式错误，请使用 '开始-结束' 格式，如 '1-5'")
        return delete_ranges
    else:
        print("无效选择，默认使用选项1")
        return [(1, 7), (58, 76), (120, 128), (170, 213), (225, 242)]


def get_num_classes_from_user():
    while True:
        try:
            num_classes = int(input("请输入分类数量 (例如：输入5则分为1-5类): ").strip())
            if num_classes > 0:
                return num_classes
            else:
                print("请输入大于0的数字")
        except ValueError:
            print("请输入有效的数字")


def get_input_output_paths():
    print("请输入绝对路径：")
    input_img_dir = input("影像文件夹路径 (例如: E:\\指纹光谱\\ZhiWen\\影像图): ").strip()
    output_dir = input("输出表格路径 (例如: E:\\指纹光谱\\ZhiWen\\合并表): ").strip()
    return input_img_dir, output_dir


def check_pil_installation():
    try:
        from PIL import Image
        return True
    except ImportError:
        return False


def create_temp_dir_on_d_drive():
    """在D盘创建临时目录"""
    d_temp_base = r"D:\temp_csv_processing"
    os.makedirs(d_temp_base, exist_ok=True)

    # 创建带时间戳的子目录以避免冲突
    import time
    timestamp = int(time.time())
    temp_dir = os.path.join(d_temp_base, f"temp_{timestamp}")
    os.makedirs(temp_dir, exist_ok=True)

    return temp_dir


def cleanup_temp_dir(temp_dir):
    """清理临时目录"""
    try:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            print(f"🗑️ 临时目录已清理: {temp_dir}")
    except Exception as e:
        print(f"⚠️ 清理临时目录失败: {e}")


def main(input_img_dir, output_dir, filename, delete_ranges=None, num_classes=5, adjust_sizes=True):
    # 检查PIL是否已安装
    if not check_pil_installation():
        print("警告: 未检测到PIL库，将无法处理影像文件")
        print("请运行: pip install Pillow")
        print("如果只有CSV文件，可以继续处理")

        # 检查是否有CSV文件
        csv_files = glob(os.path.join(input_img_dir, "*.csv"))
        if not csv_files:
            print("错误: 没有找到任何CSV文件，无法继续处理")
            return False

    # 创建D盘临时目录
    temp_dir = create_temp_dir_on_d_drive()
    print(f"📁 临时目录创建在: {temp_dir}")

    try:
        # 可选的删除波段操作
        if delete_ranges is not None:
            success = delete_specific_bands(input_img_dir, delete_ranges)
            if not success:
                print("删除波段文件时发生错误，继续处理...")

        temp_csv_dir_02 = os.path.join(temp_dir, "step02_csv")
        os.makedirs(temp_csv_dir_02, exist_ok=True)

        # 检查输入目录中的文件类型
        image_files = []
        supported_extensions = ['.tif', '.tiff', '.img']
        for ext in supported_extensions:
            image_files.extend(glob(os.path.join(input_img_dir, f'*{ext}')))

        if image_files:
            # 如果有影像文件，进行影像处理流程
            print("检测到影像文件，开始影像处理...")
            if not batch_process_images(input_img_dir, temp_csv_dir_02):
                print("影像处理失败，尝试直接处理CSV文件...")
                # 如果影像处理失败，复制原始文件或尝试其他处理方式
                pass

        # 检查是否有CSV文件
        csv_files = glob(os.path.join(input_img_dir, "*.csv"))
        if csv_files:
            # 如果有CSV文件，复制到临时目录
            for csv_file in csv_files:
                import shutil
                shutil.copy(csv_file, temp_csv_dir_02)

        temp_csv_dir_03 = os.path.join(temp_dir, "step03_csv")
        os.makedirs(temp_csv_dir_03, exist_ok=True)

        # 检查temp_csv_dir_02是否有CSV文件
        existing_csv_files = glob(os.path.join(temp_csv_dir_02, "*.csv"))
        if existing_csv_files:
            if not batch_process_csv(temp_csv_dir_02, temp_csv_dir_03):
                return False
        else:
            # 如果没有CSV文件，尝试处理原影像目录中的CSV
            original_csv_files = glob(os.path.join(input_img_dir, "*.csv"))
            if original_csv_files:
                for csv_file in original_csv_files:
                    import shutil
                    shutil.copy(csv_file, temp_csv_dir_03)
            else:
                print("未找到任何影像或CSV文件")
                return False

        temp_csv_dir_04 = os.path.join(temp_dir, "step04_csv")
        os.makedirs(temp_csv_dir_04, exist_ok=True)
        if not process_large_csv_files_with_progress(temp_csv_dir_03, temp_csv_dir_04, num_classes):
            return False

        success = merge_csv_cells_by_position(temp_csv_dir_04, output_dir, filename, adjust_sizes=adjust_sizes)
        return success

    finally:
        # 程序结束后自动清理临时目录
        cleanup_temp_dir(temp_dir)


if __name__ == "__main__":
    # 检查PIL安装
    if not check_pil_installation():
        print("警告: 未检测到PIL库，将无法处理影像文件")
        print("请运行: pip install Pillow")
        print("如果只有CSV文件，可以继续处理")

    # 获取用户输入的绝对路径
    input_img_dir, output_dir = get_input_output_paths()
    filename = input("请输入输出文件名 (不需要扩展名): ").strip()

    if not os.path.isdir(input_img_dir):
        print(f"错误: 影像文件夹不存在 → {input_img_dir}")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    # 获取用户设置
    delete_ranges = get_delete_ranges_from_user()
    num_classes = get_num_classes_from_user()

    # 询问是否调整不同尺寸的CSV文件
    print("\n=== CSV文件合并设置 ===")
    print("功能：将多个 CSV 文件中相同位置的数值合并，用空格分隔\n")
    size_adjust_choice = input("是否自动调整不同尺寸的CSV文件？(y/n，默认为y): ").strip().lower()
    adjust_sizes = True if size_adjust_choice != 'n' else False

    print("正在处理数据...")
    if main(input_img_dir, output_dir, filename, delete_ranges, num_classes, adjust_sizes):
        output_path = os.path.join(output_dir, filename + '.csv')
        print(f"✅ 处理完成！")
        print(f"📁 最终合并文件已保存至: {output_path}")
    else:
        print("❌ 处理失败")
        sys.exit(1)