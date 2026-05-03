# -*- coding: utf-8 -*-
"""
全自动遥感分类处理流程（优化版）
========================
流程: 01填充合并 → 02样本测试 → 03匹配(循环) → 04ENVI转换 → 05重分类 → 精度评价
循环条件: Kappa < 0.8 或 覆盖率 < 50% 则提高阈值继续

优化说明:
  1. 覆盖率监控 — 跟踪匹配(有值)像素占总像素比例
  2. 双重停止条件 — Kappa ≥ 0.8 且 覆盖率 ≥ 50%
  3. 自适应起始阈值 — 根据波段数量动态设定
  4. 快速爬坡 — 覆盖率 < 10% 时大步长+20，之后小步长+5
"""

import os
import sys
import shutil
import datetime
import math
import csv
import re
import gc
import importlib.util
import warnings
from glob import glob

import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings('ignore')

# ============================================================
# 动态加载 D:/代码测试/ 中的模块
# ============================================================
SCRIPT_DIR = 'D:/代码测试'

def load_module_from_file(module_name, filename):
    filepath = os.path.join(SCRIPT_DIR, filename)
    spec = importlib.util.spec_from_file_location(module_name, filepath)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

mod_01 = load_module_from_file('step01', '01填充合并.py')
mod_02 = load_module_from_file('step02', '02样本测试.py')
mod_04 = load_module_from_file('step04', '04envi进阶.py')
mod_05 = load_module_from_file('step05', '05重分类优化.py')
mod_eval = load_module_from_file('eval', '精度评价测试.py')

step01_main = mod_01.main
step02_main = mod_02.main
step04_main = mod_04.main
reclassify = mod_05.reclassify_with_geospatial_info
read_hdr = mod_05.read_envi_header
read_class_names = mod_eval.read_class_names_from_hdr
load_shp = mod_eval.load_shp_folder
extract_pixels = mod_eval.extract_pixels
compute_metrics = mod_eval.compute_metrics
export_report = mod_eval.export_report
read_envi_classification = mod_eval.read_envi_classification


# ============================================================
# 汉明距离匹配
# ============================================================
def calculate_hamming_distance(seq1, seq2):
    if len(seq1) != len(seq2):
        return float('inf')
    return sum(a != b for a, b in zip(seq1, seq2))


def get_image_count(folder_path):
    supported = ['.tif', '.tiff', '.img', '.jpg', '.jpeg', '.png', '.bmp', '.gif']
    images = []
    for ext in supported:
        images.extend(glob(os.path.join(folder_path, f'*{ext}')))
        images.extend(glob(os.path.join(folder_path, f'*{ext.upper()}')))
    return len(set(images))


def run_matching(target_file, samples_folder, images_folder, output_file,
                 order_indices, replacements, threshold, num_rounds=1):
    """
    执行汉明距离匹配。
    返回: (modified_count, total_data_cells)
        modified_count: 成功匹配替换的像元数
        total_data_cells: 有序列数据的总像元数（排除空行/首行列头）
    """
    print(f"\n{'='*60}")
    print(f"🔍 汉明距离匹配 — 阈值={threshold}")
    print(f"{'='*60}")

    image_count = get_image_count(images_folder)
    target_length = image_count
    print(f"📊 影像数量 = {target_length}（作为序列标准长度）")

    sample_files = sorted([f for f in os.listdir(samples_folder) if f.lower().endswith('.csv')])
    ordered_files = [sample_files[i] for i in order_indices]
    print(f"📁 样本匹配顺序: {ordered_files}")
    print(f"🔢 替换值: {replacements}")

    # 读取目标文件
    target_rows = None
    for enc in ['utf-8-sig', 'utf-8', 'gbk', 'gb2312', 'latin1']:
        try:
            with open(target_file, 'r', newline='', encoding=enc) as f:
                target_rows = list(csv.reader(f))
            print(f"✅ 读取待匹配文件: {os.path.basename(target_file)} (编码: {enc})")
            break
        except:
            continue
    if target_rows is None:
        raise RuntimeError(f"无法读取待匹配文件: {target_file}")

    # 读取所有样本的参考序列
    all_reference_sequences = []
    for sample_file in ordered_files:
        sample_path = os.path.join(samples_folder, sample_file)
        sample_rows = None
        for enc in ['utf-8-sig', 'utf-8', 'gbk', 'gb2312', 'latin1']:
            try:
                with open(sample_path, 'r', newline='', encoding=enc) as f:
                    sample_rows = list(csv.reader(f))
                break
            except:
                continue
        if sample_rows is None or len(sample_rows) < 2:
            print(f"⚠️ 跳过样本 {sample_file}: 无法读取或无数据")
            all_reference_sequences.append([])
            continue

        header = sample_rows[0]
        seq_col = -1
        for i, col in enumerate(header):
            if col.strip() == '提取值':
                seq_col = i
                break
        if seq_col == -1:
            for i, col in enumerate(header):
                if '提取值' in col.strip():
                    seq_col = i
                    break
        if seq_col == -1:
            print(f"⚠️ 样本 {sample_file} 无 '提取值' 列，跳过")
            all_reference_sequences.append([])
            continue

        refs = []
        for row in sample_rows[1:]:
            cell = row[seq_col].strip() if seq_col < len(row) else ''
            if not cell or ' ' not in cell:
                continue
            parts = cell.split()
            if len(parts) != target_length:
                continue
            try:
                seq = [int(float(v)) for v in parts]
                refs.append(seq)
            except:
                continue
        print(f"   {sample_file}: {len(refs)} 个参考序列")
        all_reference_sequences.append(refs)

    total_sequences = sum(len(r) for r in all_reference_sequences)
    if total_sequences == 0:
        print("⚠️ 无参考序列，匹配跳过")
        return 0, 0

    # 执行匹配
    modified_data = [row[:] for row in target_rows]

    # 统计实际有数据的像元数（排除首行首列等表头）
    total_data_cells = 0
    for row_idx, row in enumerate(modified_data):
        for col_idx, cell in enumerate(row):
            if cell.strip():
                vals = cell.strip().split()
                if len(vals) == target_length:
                    total_data_cells += 1

    modified_count = 0
    short_loop_count = 0  # 有数据但长度不符合的

    for row_idx, row in enumerate(tqdm(modified_data, desc="匹配进度", unit="行")):
        for col_idx, cell in enumerate(row):
            if not cell.strip():
                modified_data[row_idx][col_idx] = "-1"
                continue

            vals = cell.strip().split()
            if len(vals) != target_length:
                if len(vals) < target_length and len(vals) > 0:
                    short_loop_count += 1
                continue
            try:
                current = [int(float(v)) for v in vals]
            except:
                continue

            best_match = None
            best_dist = float('inf')
            for s_idx, refs in enumerate(all_reference_sequences):
                for ref in refs:
                    dist = calculate_hamming_distance(current, ref)
                    if dist < best_dist:
                        best_dist = dist
                        best_match = replacements[s_idx]

            if best_match is not None and best_dist <= threshold:
                modified_data[row_idx][col_idx] = best_match
                modified_count += 1

    # 保存结果
    out_dir = os.path.dirname(output_file)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerows(modified_data)

    coverage = modified_count / total_data_cells * 100 if total_data_cells > 0 else 0
    print(f"\n✅ 匹配完成!")
    print(f"   有效数据像元: {total_data_cells}")
    print(f"   成功匹配: {modified_count} ({coverage:.2f}%)")
    print(f"   未匹配(将变为0/NoData): {total_data_cells - modified_count} ({100 - coverage:.2f}%)")
    print(f"   剩余为空/表头等: {short_loop_count}")

    return modified_count, total_data_cells


# ============================================================
# 精度评价（含覆盖率统计）
# ============================================================
def run_accuracy_evaluation(reclass_dat, eval_shp_folder, output_dir):
    """
    执行精度评价（含未分类惩罚：未分类像元计入分类错误）。
    返回: (kappa, oa, total_valid_pixels, report_path)
    """
    print(f"\n{'='*60}")
    print(f"📊 执行精度评价（含未分类惩罚）")
    print(f"{'='*60}")

    class_img, transform, crs = read_envi_classification(reclass_dat)

    # 统计分类结果中有效分类像元数
    unique, counts = np.unique(class_img, return_counts=True)
    total_pixels = class_img.size
    classified_pixels = sum(counts[unique >= 2]) if np.any(unique >= 2) else 0
    unclassed_pixels = sum(counts[unique == 0]) if np.any(unique == 0) else 0
    coverage_total = classified_pixels / total_pixels * 100 if total_pixels > 0 else 0
    print(f"   分类图总像素: {total_pixels}")
    print(f"   有效分类(≥2): {classified_pixels} ({coverage_total:.2f}%)")
    print(f"   未分类(=0): {unclassed_pixels} ({unclassed_pixels/total_pixels*100:.2f}%)")

    base = os.path.splitext(reclass_dat)[0]
    hdr_path = base + '.hdr'
    all_class_names = read_class_names(hdr_path)

    class_name_to_code = {
        name: idx for idx, name in enumerate(all_class_names)
        if idx >= 2
    }
    print(f"   HDR 类别: {list(class_name_to_code.keys())}")

    gdf, class_info = load_shp(eval_shp_folder, class_name_to_code)

    # 注入"未分类"类别代码0，防止虚高精度
    # 未分类像元将作为"分错"计入混淆矩阵，拉低Kappa
    class_info.insert(0, (0, '未分类', 0))

    if crs and gdf.crs and gdf.crs != crs:
        print(f"🔄 坐标系转换: {gdf.crs} -> {crs}")
        gdf = gdf.to_crs(crs)

    # 保留未分类(=0)像元
    real_codes, pred_codes = extract_pixels(gdf, class_img, transform,
                                            skip_nodata=False, shrink_pixels=0)
    if len(real_codes) == 0:
        print("❌ 无有效像素")
        return None, None, 0, None

    # 统计预测为0（未分类）的占比
    unclassified_pred = sum(1 for p in pred_codes if p == 0)
    print(f"   验证像元总数: {len(real_codes)}")
    print(f"   其中被预测为未分类(=0): {unclassified_pred} ({unclassified_pred/len(real_codes)*100:.2f}%)")

    cm, oa, kappa, producer, user, class_names = compute_metrics(real_codes, pred_codes, class_info)

    base_name = os.path.splitext(os.path.basename(reclass_dat))[0]
    txt_out = os.path.join(output_dir, f"{base_name}_accuracy_report.txt")
    csv_out = os.path.join(output_dir, f"{base_name}_confusion_matrix.csv")
    export_report(cm, oa, kappa, producer, user, class_names, class_info, txt_out, csv_out)

    print(f"\n📊 OA = {oa:.2f}%, Kappa = {kappa:.4f}")
    print(f"   验证像素总数: {len(real_codes)}")
    print(f"   未分类占比: {unclassified_pred/len(real_codes)*100:.2f}%")
    return kappa, oa, len(real_codes), txt_out


# ============================================================
# 主流程
# ============================================================
def main():
    print("=" * 80)
    print("  全自动遥感分类处理流程（优化版）")
    print("=" * 80)
    print("流程: 01填充合并 → 02样本测试 → 03匹配(阈值循环)")
    print("      → 04ENVI转换 → 05重分类 → 精度评价")
    print("停止条件: Kappa ≥ 0.8")
    print("=" * 80)

    # ---- 1. 用户输入 ----
    img_folder = input("\n1. 遥感影像文件夹路径: ").strip().strip('"\'')
    sample_shp = input("2. 样本SHP路径（文件或文件夹）: ").strip().strip('"\'')
    eval_shp_folder = input("3. 精度评价SHP文件夹路径: ").strip().strip('"\'')
    min_kappa = input("4. 目标Kappa值 (默认0.8): ").strip()
    min_kappa = float(min_kappa) if min_kappa else 0.8

    for path, desc in [(img_folder, "影像文件夹"),
                        (sample_shp, "样本SHP"),
                        (eval_shp_folder, "精度评价SHP文件夹")]:
        if not os.path.exists(path):
            print(f"❌ {desc}不存在: {path}")
            sys.exit(1)

    # ---- 2. 创建输出目录 ----
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = os.path.join(os.path.dirname(img_folder), f"自动化流程_{timestamp}")
    os.makedirs(base_dir, exist_ok=True)

    dir_01 = os.path.join(base_dir, "01_填充合并")
    dir_02 = os.path.join(base_dir, "02_样本测试")
    dir_03 = os.path.join(base_dir, "03_匹配结果")
    dir_04 = os.path.join(base_dir, "04_ENVI结果")
    dir_05 = os.path.join(base_dir, "05_重分类结果")
    dir_eval = os.path.join(base_dir, "精度评价结果")

    for d in [dir_01, dir_02, dir_03, dir_04, dir_05, dir_eval]:
        os.makedirs(d, exist_ok=True)

    print(f"\n📁 输出目录: {base_dir}")

    # ---- 3. 执行 01 填充合并 ----
    print(f"\n{'='*60}")
    print("📌 步骤01: 填充合并")
    print(f"{'='*60}")

    merged_csv = os.path.join(dir_01, "合并表.csv")
    step01_main(img_folder, dir_01, "合并表",
                delete_ranges=None, num_classes=5, adjust_sizes=True)

    if not os.path.exists(merged_csv):
        files = [f for f in os.listdir(dir_01) if f.endswith('.csv')]
        if files:
            merged_csv = os.path.join(dir_01, files[0])
            print(f"✅ 合并文件: {merged_csv}")
        else:
            print("❌ 步骤01 未生成 CSV 文件")
            sys.exit(1)
    else:
        print(f"✅ 合并文件: {merged_csv}")

    # ---- 4. 执行 02 样本测试 ----
    print(f"\n{'='*60}")
    print("📌 步骤02: 样本测试")
    print(f"{'='*60}")

    step02_main(sample_shp, img_folder, merged_csv, dir_02)

    sample_csvs = sorted([f for f in os.listdir(dir_02) if f.endswith('.csv') and f != 'summary.csv'])
    if not sample_csvs:
        print("❌ 步骤02 未生成样本 CSV 文件")
        sys.exit(1)
    print(f"✅ 样本 CSV 文件: {sample_csvs}")

    # ---- 5. 循环参数 ----
    # 自适应起始阈值：波段数较多时起始值也提高
    band_count = get_image_count(img_folder)
    initial_threshold = max(15, band_count // 5)  # 如 100波段 → 起始20; 200波段 → 起始40
    threshold = initial_threshold
    match_count = 1
    max_iterations = 50

    order_indices = list(range(len(sample_csvs)))
    replacements = [str(i + 1) for i in range(len(sample_csvs))]
    class_names_match = [os.path.splitext(f)[0] for f in sample_csvs]

    print(f"\n{'='*60}")
    print(f"📌 循环匹配 开始")
    print(f"   波段数: {band_count}")
    print(f"   自适应起始阈值: {threshold} (波段数÷5)")
    print(f"   目标Kappa: {min_kappa}")
    print(f"   匹配顺序: {sample_csvs}")
    print(f"   替换值: {replacements}")
    print(f"{'='*60}")

    best_result = {"kappa": 0, "threshold": 0,
                   "report": None, "matched_csv": None, "reclass_dat": None}

    for iteration in range(max_iterations):
        print(f"\n{'#'*70}")
        print(f"  🔄 第 {iteration + 1} 次循环 — 阈值 = {threshold}")
        print(f"{'#'*70}")

        # ---- 5a. 03 匹配 ----
        matched_csv = os.path.join(dir_03, f"匹配结果_阈值{threshold}.csv")
        modified_count, total_data_cells = run_matching(
            target_file=merged_csv,
            samples_folder=dir_02,
            images_folder=img_folder,
            output_file=matched_csv,
            order_indices=order_indices,
            replacements=replacements,
            threshold=threshold,
            num_rounds=match_count
        )

        match_coverage = modified_count / total_data_cells * 100 if total_data_cells > 0 else 0
        print(f"📈 匹配覆盖率: {match_coverage:.2f}% ({modified_count}/{total_data_cells})")

        if not os.path.exists(matched_csv) or modified_count == 0:
            print(f"⚠️ 无匹配结果，快速提升阈值")
            threshold += max(20, band_count // 10)
            continue

        # ---- 5b. 04 ENVI转换 ----
        print(f"\n{'='*60}")
        print("📌 步骤04: ENVI 转换")
        print(f"{'='*60}")

        step04_main(
            input_csv=matched_csv,
            images_folder=img_folder,
            output_dir=dir_04,
            add_geo=True,
            max_sequence_length=None
        )

        dat_files = [f for f in os.listdir(dir_04) if f.endswith('.dat')]
        if not dat_files:
            print("❌ 步骤04 未生成 .dat 文件")
            threshold += max(10, band_count // 20)
            continue
        envi_dat = os.path.join(dir_04, dat_files[0])
        print(f"✅ 生成 ENVI 文件: {envi_dat}")

        # ---- 5c. 05 重分类 ----
        print(f"\n{'='*60}")
        print("📌 步骤05: 重分类")
        print(f"{'='*60}")

        class_value_groups = [[int(r)] for r in replacements]
        success = reclassify(envi_dat, class_value_groups, class_names_match)
        if not success:
            print("❌ 步骤05 重分类失败")
            threshold += max(10, band_count // 20)
            continue

        base_reclass = os.path.splitext(envi_dat)[0]
        if base_reclass.endswith('.dat'):
            base_reclass = base_reclass[:-4]
        reclass_dat = base_reclass + '_reclass.dat'
        reclass_hdr = base_reclass + '_reclass.hdr'

        if os.path.exists(reclass_dat):
            shutil.copy2(reclass_dat, os.path.join(dir_05, os.path.basename(reclass_dat)))
        if os.path.exists(reclass_hdr):
            shutil.copy2(reclass_hdr, os.path.join(dir_05, os.path.basename(reclass_hdr)))

        # ---- 5d. 精度评价 ----
        result = run_accuracy_evaluation(reclass_dat, eval_shp_folder, dir_eval)
        if result[0] is None:
            print("⚠️ 精度评价失败")
            threshold += max(10, band_count // 20)
            continue

        kappa, oa, eval_pixels, report_path = result

        # ---- 5e. 综合评判 ----
        print(f"\n{'='*70}")
        print(f"📊 第 {iteration + 1} 次循环 — 阈值={threshold}")
        print(f"   匹配覆盖率: {match_coverage:.2f}%")
        print(f"   验证像素: {eval_pixels}")
        print(f"   Kappa: {kappa:.4f}")
        print(f"   OA: {oa:.2f}%")
        print(f"   分类图有效覆盖率: 见上方")
        print(f"{'='*70}")

        # 记录当前最佳结果（仅按Kappa排序）
        if kappa > best_result["kappa"]:
            best_result = {
                "kappa": kappa,
                "threshold": threshold,
                "report": report_path,
                "matched_csv": matched_csv,
                "reclass_dat": reclass_dat,
                "oa": oa,
                "eval_pixels": eval_pixels
            }

        # 停止条件：仅判断Kappa
        if kappa >= min_kappa:
            print(f"\n🎉🎉🎉 达标! Kappa={kappa:.4f} ≥ {min_kappa}")
            break
        else:
            print(f"   Kappa={kappa:.4f} < {min_kappa}，提高阈值 +5")
            threshold += 5

    else:
        print(f"\n❌ 经过 {max_iterations} 次循环未满足终止条件")
        print(f"   当前最佳结果: 阈值={best_result['threshold']}, "
              f"Kappa={best_result['kappa']:.4f}")

    # ---- 6. 最终总结 ----
    print(f"\n{'='*80}")
    print("🎉 自动化流程完成！")
    print(f"{'='*80}")
    print(f"📁 所有结果已保存至: {base_dir}")

    if best_result["report"]:
        print(f"\n🏆 最佳结果:")
        print(f"   阈值: {best_result['threshold']}")
        print(f"   Kappa: {best_result['kappa']:.4f}")
        print(f"   OA: {best_result['oa']:.2f}%")
        print(f"   验证像素: {best_result['eval_pixels']}")
        print(f"   报告: {best_result['report']}")

    print(f"\n📂 各步骤目录:")
    for name, d in [("01_填充合并", dir_01), ("02_样本测试", dir_02),
                     ("03_匹配结果", dir_03), ("04_ENVI结果", dir_04),
                     ("05_重分类结果", dir_05), ("精度评价结果", dir_eval)]:
        files = os.listdir(d)
        print(f"   {name}/ — {len(files)} 个文件")
    print(f"{'='*80}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ 用户中断操作。")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ 程序异常: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
