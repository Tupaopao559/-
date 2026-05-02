import os
import csv
from tqdm import tqdm
import re
from glob import glob


def calculate_hamming_distance(seq1, seq2):
    """计算两个序列的汉明距离"""
    if len(seq1) != len(seq2):
        return float('inf')
    return sum(a != b for a, b in zip(seq1, seq2))


def get_image_count_from_folder(folder_path):
    """从文件夹中统计影像图数量"""
    supported_extensions = ['.tif', '.tiff', '.img', '.jpg', '.jpeg', '.png', '.bmp', '.gif']
    image_files = []

    for ext in supported_extensions:
        image_files.extend(glob(os.path.join(folder_path, f'*{ext}')))
        image_files.extend(glob(os.path.join(folder_path, f'*{ext.upper()}')))

    # 去重
    image_files = list(set(image_files))
    return len(image_files), image_files


def multi_standard_hamming_match():
    print("=== 基于影像图数量的多标准序列汉明距离批处理工具（优化版）===\n")

    # 获取用户输入
    images_folder = input("请输入包含影像图的文件夹路径（用于确定序列标准长度）: ").strip().strip('"\'')
    samples_folder = input("请输入样本文件夹路径（包含多个地物样本CSV文件）: ").strip().strip('"\'')
    target_file = input("请输入待匹配的CSV文件路径（含序列的网格数据）: ").strip().strip('"\'')
    output_file = input("请输入输出CSV文件路径（包含文件名，如：output.csv）: ").strip().strip('"\'')

    # 获取影像图数量以确定标准长度
    image_count, image_files = get_image_count_from_folder(images_folder)
    if image_count == 0:
        print("❌ 在影像图文件夹中未找到任何支持的影像文件")
        print("支持的格式: .tif, .tiff, .img, .jpg, .jpeg, .png, .bmp, .gif")
        return False

    print(f"\n📊 检测到 {image_count} 个影像文件，将使用 {image_count} 作为序列标准长度")
    print("影像文件列表:")
    for i, img_file in enumerate(image_files[:5]):  # 只显示前5个
        print(f"  {i + 1}: {os.path.basename(img_file)}")
    if len(image_files) > 5:
        print(f"  ... 还有 {len(image_files) - 5} 个文件")

    # 获取样本文件列表
    sample_files = [f for f in os.listdir(samples_folder) if f.lower().endswith('.csv')]
    if not sample_files:
        print("❌ 样本文件夹中未找到任何CSV文件")
        return False

    print(f"\n📁 检测到 {len(sample_files)} 个样本文件:")
    for i, file in enumerate(sample_files):
        print(f"  {i + 1}: {file}")

    # 获取匹配顺序
    order_input = input(f"\n请输入匹配顺序（如：132 表示先匹配第1个，再匹配第3个，最后匹配第2个）: ").strip()
    try:
        order_indices = [int(digit) - 1 for digit in order_input]
        if any(idx < 0 or idx >= len(sample_files) for idx in order_indices):
            print("❌ 匹配顺序超出范围")
            return False
        ordered_files = [sample_files[i] for i in order_indices]
    except ValueError:
        print("❌ 匹配顺序格式错误，请输入数字")
        return False

    # 获取匹配轮次
    num_rounds = int(input("请输入汉明距离匹配次数: ").strip())

    # 获取每轮的阈值和替换值
    rounds_config = []
    for i in range(num_rounds):
        print(f"\n--- 第 {i + 1} 轮匹配设置 ---")
        threshold = int(input(f"请输入第 {i + 1} 轮汉明距离阈值: ").strip())

        # 获取该轮每个样本的替换值
        round_replacements = []
        for j, sample_file in enumerate(ordered_files):
            replacement = input(f"  为样本 '{sample_file}' 设置替换值: ").strip()
            round_replacements.append(replacement)

        rounds_config.append({
            'threshold': threshold,
            'replacements': round_replacements
        })

    # 确保输出目录存在
    output_dir = os.path.dirname(output_file) or '.'
    os.makedirs(output_dir, exist_ok=True)

    # ========== 步骤1：读取待匹配的 target_file ==========
    target_rows = None
    encodings = ['utf-8-sig', 'utf-8', 'gbk', 'gb2312', 'latin1']

    for encoding in encodings:
        try:
            with open(target_file, 'r', newline='', encoding=encoding) as f:
                target_rows = list(csv.reader(f))
            print(f"✅ 成功读取待匹配文件，使用编码: {encoding}")
            break
        except Exception as e:
            print(f"尝试编码 {encoding} 读取待匹配文件失败: {e}")

    if target_rows is None:
        print("❌ 无法读取待匹配文件")
        return False

    # 使用影像图数量作为序列标准长度
    target_sequence_length = image_count
    print(f"📊 使用影像图数量作为序列标准长度: {target_sequence_length}")

    # 显示目标文件中的一些示例序列
    example_sequences = []
    for row in target_rows:
        for cell in row:
            if cell.strip():
                values = cell.split()
                if len(values) == target_sequence_length:
                    example_sequences.append(cell[:100])
                    if len(example_sequences) >= 3:  # 只显示前3个示例
                        break
        if len(example_sequences) >= 3:
            break

    if example_sequences:
        print("📊 目标文件中的示例序列:")
        for i, seq in enumerate(example_sequences):
            print(f"   {i + 1}: {seq}")

    # ========== 步骤2：从每个样本文件中提取参考序列 ==========
    all_reference_sequences = []  # 每个样本文件的参考序列列表
    sample_names = []  # 每个样本文件的名称

    for sample_file in ordered_files:
        sample_path = os.path.join(samples_folder, sample_file)
        sample_names.append(sample_file)

        # 读取样本文件
        sample_rows = None
        for encoding in encodings:
            try:
                with open(sample_path, 'r', newline='', encoding=encoding) as f:
                    sample_rows = list(csv.reader(f))
                print(f"✅ 成功读取样本文件: {sample_file}，使用编码: {encoding}")
                break
            except Exception as e:
                print(f"尝试编码 {encoding} 读取样本文件 {sample_file} 失败: {e}")

        if sample_rows is None:
            print(f"❌ 无法读取样本文件: {sample_file}")
            continue

        # 检查文件内容结构
        if len(sample_rows) < 2:
            print(f"⚠️ 样本文件 {sample_file} 内容为空或只有表头")
            continue

        header_sample = sample_rows[0]
        print(f"📊 {sample_file} 的列名: {header_sample}")

        # 寻找'提取值'列 - 修复版逻辑
        sequence_column_idx = -1

        # 首先查找精确匹配的'提取值'列
        for i, col_name in enumerate(header_sample):
            if col_name.strip() == '提取值':
                sequence_column_idx = i
                break

        # 如果没找到精确匹配，再查找其他可能的列名
        if sequence_column_idx == -1:
            for i, col_name in enumerate(header_sample):
                clean_name = col_name.strip()
                if '提取值' in clean_name:
                    sequence_column_idx = i
                    break

        # 如果还是没找到，提示用户
        if sequence_column_idx == -1:
            print(f"❌ 在样本文件 {sample_file} 中未找到'提取值'列")
            print(f"   可用列名: {header_sample}")
            continue
        else:
            print(f"🔍 找到'提取值'列: '{header_sample[sequence_column_idx]}' (第{sequence_column_idx + 1}列)")

        reference_sequences = []
        valid_rows_count = 0
        invalid_rows_count = 0

        for i in range(1, len(sample_rows)):
            cell = sample_rows[i][sequence_column_idx].strip()
            if not cell:
                continue

            # 检查是否是单个数值（如3389.0）
            try:
                float_val = float(cell)
                if cell.count(' ') == 0:  # 真正的单个数值，不是序列
                    print(f"⚠️ 跳过单个数值（非序列）: {cell}")
                    continue
            except ValueError:
                pass  # 不是数值，可能是序列

            try:
                values = cell.split()
                if len(values) == target_sequence_length:  # 使用影像图数量作为标准长度
                    # 尝试转换为整数，处理可能的浮点数
                    try:
                        seq = [int(float(v)) for v in values]  # 先转浮点再转整数，处理"3.0"这种情况
                    except ValueError:
                        # 如果无法转换，跳过
                        print(f"⚠️ 无法转换序列: {cell[:60]}...")
                        continue
                    reference_sequences.append(seq)
                    valid_rows_count += 1
                    # 只显示第一个有效序列作为示例
                    if valid_rows_count == 1:
                        print(f"   📝 示例序列: {cell[:100]}")
                elif len(values) == 1:
                    # 真正的单个数值，跳过
                    print(f"⚠️ 跳过单个数值（非序列）: {cell}")
                    continue
                else:
                    # 长度不匹配的序列，记录但不加入参考序列
                    print(f"⚠️ 跳过长度不匹配的序列（长度={len(values)}，标准={target_sequence_length}）: {cell[:60]}...")
                    invalid_rows_count += 1
            except Exception as e:
                print(f"⚠️ 解析参考序列失败: {cell} → {e}")

        print(f"✅ 从 {sample_file} 中提取 {len(reference_sequences)} 个参考序列")
        print(f"   有效行数: {valid_rows_count}, 无效行数: {invalid_rows_count}")

        # 如果当前样本文件没有有效的参考序列，给出提示
        if len(reference_sequences) == 0:
            print(f"⚠️ {sample_file} 中没有找到有效的参考序列")
            print("   请检查：")
            print(f"   - '提取值'列是否包含长度为 {target_sequence_length} 的序列")
            print("   - 序列是否为数字，以空格分隔")
            print("   - 示例格式: 1 2 1 3 1 3 4 2 3 2...")

        all_reference_sequences.append(reference_sequences)

    # 检查是否有有效的参考序列
    total_sequences = sum(len(sequences) for sequences in all_reference_sequences)
    if total_sequences == 0:
        print("❌ 所有样本文件中都没有找到与标准长度匹配的参考序列")
        print(f"请检查样本文件的'提取值'列是否包含长度为 {target_sequence_length} 的序列数据")
        print("注意：序列应为以空格分隔的数字，如：1 2 1 3 1 3 4 2 3 2...")
        print(f"标准长度由影像图数量确定: {target_sequence_length} (来自 {image_count} 张影像图)")
        return False

    print(f"📊 总共找到 {total_sequences} 个有效参考序列")

    # ========== 步骤3：多轮匹配 ==========
    modified_data = [row[:] for row in target_rows]  # 复制原始数据

    total_cells = sum(len(row) for row in target_rows if row)
    total_modified = 0
    total_empty_assigned = 0  # 统计被赋予-1的空单元格数量

    for round_num in range(num_rounds):
        print(f"\n🔄 开始第 {round_num + 1} 轮匹配...")
        current_threshold = rounds_config[round_num]['threshold']
        current_replacements = rounds_config[round_num]['replacements']

        modified_in_round = 0
        empty_assigned_in_round = 0

        with tqdm(total=total_cells, desc=f"第{round_num + 1}轮匹配进度", unit="单元格") as pbar:
            for row_idx, row in enumerate(modified_data):
                for col_idx, cell in enumerate(row):
                    # 检查单元格是否为空（仅空白字符）
                    if not cell.strip():  # 空白或纯空格
                        modified_data[row_idx][col_idx] = "-1"
                        empty_assigned_in_round += 1
                        total_empty_assigned += 1
                        pbar.update(1)
                        continue

                    # 单元格非空，开始处理
                    try:
                        vals = cell.split()

                        if len(vals) != target_sequence_length:
                            # 长度不匹配，保持原值
                            pbar.update(1)
                            continue

                        try:
                            current = [int(float(v)) for v in vals]
                        except ValueError:
                            # 无法转换为整数，保持原值
                            pbar.update(1)
                            continue

                        best_match = None
                        best_distance = float('inf')

                        for sample_idx, ref_sequences in enumerate(all_reference_sequences):
                            for ref in ref_sequences:
                                dist = calculate_hamming_distance(current, ref)
                                if dist < best_distance:
                                    best_distance = dist
                                    best_match = current_replacements[sample_idx]

                        if best_match is not None and best_distance <= current_threshold:
                            modified_data[row_idx][col_idx] = best_match
                            modified_in_round += 1
                            total_modified += 1

                        pbar.update(1)
                    except Exception as e:
                        # 其他异常，保持原值
                        pbar.update(1)

        print(
            f"✅ 第 {round_num + 1} 轮匹配完成，修改了 {modified_in_round} 个单元格，空单元格赋值-1: {empty_assigned_in_round} 个")

        # === 保存本轮中间结果 ===
        round_output_file = os.path.join(output_dir, f"out{round_num + 1}_{current_threshold}.csv")
        try:
            with open(round_output_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerows(modified_data)
            print(f"💾 第 {round_num + 1} 轮结果已保存至: {round_output_file}")
        except Exception as e:
            print(f"⚠️ 保存第 {round_num + 1} 轮结果失败: {e}")

    # ========== 步骤4：保存最终结果 ==========
    try:
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerows(modified_data)
    except PermissionError:
        print(f"❌ 无法写入文件: {output_file}")
        print("请检查：")
        print("1. 文件路径是否正确")
        print("2. 文件是否被其他程序打开")
        print("3. 是否有写入权限")
        return False

    print(f"\n✅ 最终结果已保存至: {output_file}")
    print("\n" + "=" * 60)
    print("📊 处理总结:")
    print(f"- 影像图文件夹: {images_folder}")
    print(f"- 影像图数量: {image_count} (作为序列标准长度)")
    print(f"- 样本文件夹: {samples_folder}")
    print(f"- 待匹配文件: {os.path.basename(target_file)}")
    print(f"- 样本文件顺序: {ordered_files}")
    print(f"- 匹配轮次: {num_rounds}")
    print(f"- 序列标准长度: {target_sequence_length} (基于影像图数量)")

    for i, config in enumerate(rounds_config):
        print(f"- 第{i + 1}轮 - 阈值: {config['threshold']}, 替换值: {config['replacements']}")

    print(f"- 总单元格: {total_cells}")
    print(f"- 总参考序列: {total_sequences}")
    print(f"- 匹配成功单元格: {total_modified}")
    print(f"- 空单元格赋值-1: {total_empty_assigned}")
    print("-1值含义:")
    print("  - 单元格内容为空（仅空白字符）")
    print("注意: 长度不匹配或转换失败的单元格保持原始值不变")
    print("=" * 60)

    return True


if __name__ == "__main__":
    try:
        multi_standard_hamming_match()
    except KeyboardInterrupt:
        print("\n\n⚠️ 用户中断操作。")
    except Exception as e:
        print(f"\n❌ 程序异常: {e}")
        import traceback
        traceback.print_exc()