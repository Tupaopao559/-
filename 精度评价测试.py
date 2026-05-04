# -*- coding: utf-8 -*-
"""
ENVI 分类精度评价（基于 SHP 文件夹 + 名称匹配）
- 用户输入重分类结果文件（_reclass.dat），自动从对应 .hdr 中读取类别名称
- 读取 SHP 文件夹，根据 SHP 文件名（如 耕地.shp）与 HDR 中的 class names 匹配
- 自动分配正确的类别代码，输出混淆矩阵、OA、Kappa、用户/生产者精度
"""

import os
import sys
import re
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio import features
from sklearn.metrics import confusion_matrix, cohen_kappa_score, accuracy_score
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')


# ============================================================
# 1. 读取重分类结果 HDR 中的类别名称
# ============================================================

def read_class_names_from_hdr(hdr_path):
    """
    从 ENVI HDR 文件中解析 class names 字段。
    支持 UTF-8 和 GBK 两种编码（ENVI 中文版常用 GBK）。
    返回: list，如 ["Unclassified", "Border", "耕地", "林地", "水体"]
    """
    if not os.path.exists(hdr_path):
        raise FileNotFoundError(f"HDR 文件不存在: {hdr_path}")

    # 尝试 GBK 优先，失败则尝试 UTF-8
    # 注意：不能用 errors='ignore'，否则 GBK 中文会被 UTF-8 静默丢弃
    content = None
    for encoding in ['gbk', 'utf-8']:
        try:
            with open(hdr_path, 'r', encoding=encoding) as f:
                content = f.read()
            if content.strip():
                break
        except UnicodeDecodeError:
            continue

    if not content or not content.strip():
        raise ValueError(f"无法读取 HDR 文件: {hdr_path}")

    # 解析 class names = { ... }
    match = re.search(r'class names\s*=\s*\{(.+?)\}', content, re.IGNORECASE | re.DOTALL)
    if not match:
        raise ValueError(f"HDR 文件中未找到 'class names' 字段: {hdr_path}")

    names_str = match.group(1).strip()
    class_names = [name.strip() for name in names_str.split(',')]

    print(f"\n📖 从 HDR 读取类别名称（共 {len(class_names)} 个）:")
    for i, name in enumerate(class_names):
        print(f"   索引 {i}: '{name}'")
    print(f"   其中用户定义类别（跳过 Unclassified/Border）: {class_names[2:]}")

    return class_names


# ============================================================
# 2. 读取重分类结果影像
# ============================================================

def read_envi_classification(dat_path):
    """读取 ENVI 分类结果（重分类后的 _reclass.dat）"""
    if not os.path.exists(dat_path):
        if not dat_path.endswith('.dat') and os.path.exists(dat_path + '.dat'):
            dat_path += '.dat'
        else:
            raise FileNotFoundError(f"分类文件不存在: {dat_path}")

    with rasterio.open(dat_path) as src:
        data = src.read(1)
        transform = src.transform
        crs = src.crs

    print(f"\n✅ 分类影像: {os.path.basename(dat_path)}")
    print(f"   尺寸: {data.shape[1]} 列 x {data.shape[0]} 行")
    unique_vals = np.unique(data)
    print(f"   影像唯一值: {unique_vals}")
    return data, transform, crs


# ============================================================
# 3. 读取 SHP 文件夹 — 按名称匹配类别代码
# ============================================================

def load_shp_folder(shp_folder, class_name_to_code):
    """
    读取 SHP 文件夹。根据 SHP 文件名（不含扩展名）从 class_name_to_code 字典
    中查找对应的类别代码，实现按名称匹配。
    
    参数:
        shp_folder: SHP 文件夹路径
        class_name_to_code: dict，如 {"耕地": 2, "林地": 3, "水体": 4}
        
    返回: (combined_gdf, class_info)
        combined_gdf: 合并后的 GeoDataFrame，含 'true_code' 和 'class_name' 列
        class_info: [(class_code, class_name, polygon_count), ...]
    """
    if not os.path.isdir(shp_folder):
        raise NotADirectoryError(f"SHP 文件夹不存在: {shp_folder}")

    # 获取所有 SHP 文件（为清晰起见仍排序，但类别代码不再依赖排序顺序）
    shp_files = sorted([f for f in os.listdir(shp_folder) if f.lower().endswith('.shp')])
    if not shp_files:
        raise FileNotFoundError(f"SHP 文件夹中没有 .shp 文件: {shp_folder}")

    print(f"\n📁 SHP 文件夹: {os.path.basename(shp_folder)}")
    print(f"   找到 {len(shp_files)} 个 SHP 文件，按名称匹配类别...")

    class_info = []
    all_gdfs = []

    for shp_file in shp_files:
        class_name = os.path.splitext(shp_file)[0]  # 如 "耕地.shp" → "耕地"
        shp_path = os.path.join(shp_folder, shp_file)

        # 按名称查找类别代码
        if class_name not in class_name_to_code:
            available = list(class_name_to_code.keys())
            raise KeyError(
                f"\n❌ SHP 文件 '{shp_file}' 的类别名称 '{class_name}' "
                f"未在重分类结果中找到！\n"
                f"   可用类别: {available}\n"
                f"💡 请检查 SHP 文件名是否与 .hdr 中 class names 一致。"
            )

        class_code = class_name_to_code[class_name]

        gdf = gpd.read_file(shp_path)
        original_count = len(gdf)
        print(f"   '{class_name}' → 类别代码 {class_code} — {original_count} 个多边形")

        if len(gdf) == 0:
            print(f"      ⚠️ 跳过空文件")
            continue

        # 添加类别信息
        gdf['true_code'] = class_code
        gdf['class_name'] = class_name

        # 只保留有效几何
        gdf = gdf[gdf.geometry.is_valid & ~gdf.geometry.is_empty]
        valid_count = len(gdf)
        if valid_count == 0:
            print(f"      ⚠️ 无有效几何，跳过")
            continue
        if valid_count < original_count:
            print(f"      ⚠️ 过滤掉 {original_count - valid_count} 个无效几何")

        all_gdfs.append(gdf)
        class_info.append((class_code, class_name, valid_count))

    if not all_gdfs:
        raise ValueError("所有 SHP 文件均无有效多边形，无法进行精度评价")

    # 合并所有 GDF
    combined_gdf = pd.concat(all_gdfs, ignore_index=True)
    print(f"\n📊 有效多边形总计: {len(combined_gdf)} 个")
    for code, name, count in class_info:
        print(f"   类别 {code} ({name}): {count} 个多边形")

    return combined_gdf, class_info


# ============================================================
# 4. 提取像元（点提取版，直接用 rowcol 定位）
# ============================================================

from rasterio.transform import rowcol

def extract_pixels(gdf, class_img, transform, skip_nodata=True, shrink_pixels=0):
    """
    提取每个样本点对应像元的真实代码和预测代码。
    验证SHP为点文件时，直接用 rowcol 将点坐标转为行列号。
    验证SHP为多边形时，使用栅格化掩膜。

    参数:
        skip_nodata: True=跳过0和1, False=保留未分类像元计入错误
        shrink_pixels: 无用（保留兼容）
    返回: (real_codes, pred_codes)
    """
    h, w = class_img.shape
    real_codes = []
    pred_codes = []
    class_img_int = class_img.astype(np.int32)

    # 判断是点还是面
    first_geom = gdf.iloc[0].geometry
    is_point = 'Point' in first_geom.geom_type

    if is_point:
        # ---- 点文件：直接用 rowcol 定位 ----
        print("   检测到点文件，直接使用坐标定位...")
        for idx, row in tqdm(gdf.iterrows(), total=len(gdf), desc="处理验证点"):
            geom = row.geometry
            true_code = int(row['true_code'])

            # ENVI point ROIs are evaluated against the nearest pixel center.
            r, c = rowcol(transform, geom.x, geom.y, op=np.round)
            r, c = int(r), int(c)
            if 0 <= r < h and 0 <= c < w:
                val = int(class_img_int[r, c])
                if val == 1:
                    continue
                if skip_nodata and val == 0:
                    continue
                real_codes.append(true_code)
                pred_codes.append(val)
    else:
        # ---- 面文件：使用栅格化掩膜 ----
        print("   检测到面文件，使用栅格化掩膜...")
        for idx, row in tqdm(gdf.iterrows(), total=len(gdf), desc="处理验证多边形"):
            geom = row.geometry
            true_code = int(row['true_code'])

            try:
                mask = features.geometry_mask([geom], out_shape=(h, w), transform=transform, invert=True)
            except Exception as e:
                print(f"   警告: 多边形 {idx} 掩膜失败: {e}")
                continue

            pixels = class_img_int[mask]
            if skip_nodata:
                valid_mask = (pixels != 0) & (pixels != 1)
            else:
                valid_mask = pixels != 1
            valid_pixels = pixels[valid_mask]
            if len(valid_pixels) == 0:
                continue

            real_codes.extend([true_code] * len(valid_pixels))
            pred_codes.extend(valid_pixels.tolist())

    print(f"\n✅ 共提取 {len(real_codes)} 个有效像元")
    return real_codes, pred_codes


# ============================================================
# 5. 计算精度指标
# ============================================================

def compute_metrics(real_codes, pred_codes, class_info):
    """
    使用原始像素值计算混淆矩阵和精度。
    class_info: [(class_code, class_name, count), ...]
    """
    # 按 class_info 的顺序确定类别
    unique_codes = [info[0] for info in class_info]
    class_names = [info[1] for info in class_info]

    code_to_idx = {code: i for i, code in enumerate(unique_codes)}
    n_classes = len(unique_codes)

    # 转换真实标签
    real_idx = [code_to_idx[c] for c in real_codes]

    # 预测标签：如果在真实类别中则正常映射，否则映射到最近类别
    pred_idx = []
    for p in pred_codes:
        if p in code_to_idx:
            pred_idx.append(code_to_idx[p])
        else:
            nearest = min(unique_codes, key=lambda x: abs(x - p))
            pred_idx.append(code_to_idx[nearest])

    n_mapped = sum(1 for p in pred_codes if p not in code_to_idx)
    if n_mapped > 0:
        print(f"⚠️ 检测到 {n_mapped} 个预测值不在真实类别中的像素，已映射到最近类别")

    labels = list(range(n_classes))
    cm = confusion_matrix(real_idx, pred_idx, labels=labels)
    oa = accuracy_score(real_idx, pred_idx) * 100
    kappa = cohen_kappa_score(real_idx, pred_idx)

    producer = []
    user = []
    for i in range(n_classes):
        total_actual = cm[i, :].sum()
        total_pred = cm[:, i].sum()
        producer.append(cm[i, i] / total_actual * 100 if total_actual > 0 else 0)
        user.append(cm[i, i] / total_pred * 100 if total_pred > 0 else 0)

    return cm, oa, kappa, producer, user, class_names


# ============================================================
# 6. 导出报告
# ============================================================

# ============================================================
# 6. 导出报告（ENVI 标准格式）
# ============================================================

def export_report(cm, oa, kappa, producer, user, class_names, class_info, output_txt, output_csv):
    """
    导出精度评价报告（ENVI 标准格式）。
    cm: sklearn 混淆矩阵，cm[i,j] = 真实=i, 预测=j
    class_info: [(class_code, class_name, count), ...]
    """
    n = len(class_names)
    display_names = [f"{name}({code})" for code, name, _ in class_info]

    # sklearn cm[i,j] = 真实=i, 预测=j → 与 ENVI 格式一致（行=真实, 列=预测）
    # 不需要转置
    cm_display = cm

    # 各类别总计
    ref_totals = cm_display.sum(axis=1)    # 真实总数（行和）
    pred_totals = cm_display.sum(axis=0)   # 预测总数（列和）
    total_pixels = cm.sum()

    # 百分比混淆矩阵（按行归一化，每行真实类 = 100%）
    cm_pct = np.zeros_like(cm_display, dtype=float)
    for i in range(n):
        if ref_totals[i] > 0:
            cm_pct[i, :] = cm_display[i, :] / ref_totals[i] * 100
    ref_pct_total = np.full(n, 100.0)

    # 行百分比（每行预测类 = 100%）
    cm_row_pct = np.zeros_like(cm_display, dtype=float)
    for i in range(n):
        if pred_totals[i] > 0:
            cm_row_pct[i, :] = cm_display[i, :] / pred_totals[i] * 100
    pred_pct_total = np.full(n, 100.0)

    # Commission / Omission
    commission_pct = []
    omission_pct = []
    commission_pix = []
    omission_pix = []
    for i in range(n):
        # Commission: 预测为 i 但不是真实 i / 预测总数
        comm_pix = pred_totals[i] - cm_display[i, i]
        commission_pix.append(comm_pix)
        commission_pct.append(comm_pix / pred_totals[i] * 100 if pred_totals[i] > 0 else 0)

        # Omission: 真实为 i 但未被预测为 i / 真实总数
        om_pix = ref_totals[i] - cm_display[i, i]
        omission_pix.append(om_pix)
        omission_pct.append(om_pix / ref_totals[i] * 100 if ref_totals[i] > 0 else 0)

    # ====== 写入 TXT 报告 ======
    with open(output_txt, 'w', encoding='utf-8') as f:
        sep = "=" * 100
        f.write(sep + "\n")
        f.write("ENVI 分类精度评价报告\n")
        f.write(sep + "\n\n")

        # --- 1. 像素数混淆矩阵（行=真实, 列=预测） ---
        f.write("Confusion Matrix (Pixels)\n")
        f.write("                " + "".join(f"{dn:<14}" for dn in display_names) + f"{'Total':<8}\n")
        for i in range(n):
            f.write(f"{display_names[i]:<16}")
            for j in range(n):
                f.write(f"{int(cm_display[i, j]):<14}")
            f.write(f"{int(ref_totals[i]):<8}\n")
        f.write(f"{'Total':<16}")
        for j in range(n):
            f.write(f"{int(pred_totals[j]):<14}")
        f.write(f"{int(total_pixels):<8}\n")

        # --- 2. 百分比混淆矩阵（每行真实类=100%） ---
        f.write("\nConfusion Matrix (Percent)\n")
        f.write("                " + "".join(f"{dn:<14}" for dn in display_names) + f"{'Total':<8}\n")
        for i in range(n):
            f.write(f"{display_names[i]:<16}")
            for j in range(n):
                f.write(f"{cm_pct[i, j]:<14.2f}")
            f.write(f"{100.0:<8.2f}\n")
        f.write(f"{'Total':<16}")
        for j in range(n):
            total_pct = pred_totals[j] / total_pixels * 100 if total_pixels > 0 else 0
            f.write(f"{total_pct:<14.2f}")
        f.write(f"{100.0:<8.2f}\n")

        # --- 3. Commission / Omission ---
        f.write("\nClass    Commission(%)    Omission(%)    Commission(Pix)    Omission(Pix)\n")
        f.write("-" * 75 + "\n")
        for i in range(n):
            f.write(f"{display_names[i]:<10} "
                    f"{commission_pct[i]:<18.2f} "
                    f"{omission_pct[i]:<14.2f} "
                    f"{int(commission_pix[i]):<8}/{int(pred_totals[i]):<8}      "
                    f"{int(omission_pix[i]):<5}/{int(ref_totals[i]):<5}\n")

        # --- 4. 精度 ---
        f.write("\nClass    Producer(%)    User(%)        Producer(Pix)    User(Pix)\n")
        f.write("-" * 75 + "\n")
        for i in range(n):
            f.write(f"{display_names[i]:<10} "
                    f"{producer[i]:<16.2f} "
                    f"{user[i]:<14.2f} "
                    f"{int(cm_display[i, i]):<8}/{int(ref_totals[i]):<8}      "
                    f"{int(cm_display[i, i]):<5}/{int(pred_totals[i]):<5}\n")

        # --- 5. 汇总 ---
        f.write(f"\n{'Overall Accuracy:':<20} {oa:.2f}% ({int(np.diag(cm).sum())}/{int(total_pixels)})\n")
        f.write(f"{'Kappa Coefficient:':<20} {kappa:.4f}\n")

    # ====== CSV 混淆矩阵 ======
    cm_df = pd.DataFrame(cm, index=display_names, columns=display_names)
    cm_df.to_csv(output_csv, encoding='utf-8-sig')

    print(f"\n📄 报告已保存: {output_txt}")
    print(f"📊 混淆矩阵 CSV: {output_csv}")


# ============================================================
# 7. 主流程
# ============================================================

def main():
    print("=" * 80)
    print("ENVI 分类精度评价 — 基于名称匹配")
    print("=" * 80)
    print("说明：")
    print("  • 输入重分类结果文件（_reclass.dat），自动读取对应 .hdr 中的类别名称")
    print("  • 输入 SHP 文件夹，根据 SHP 文件名与类别名称自动匹配")
    print("  • 示例：SHP 文件 '耕地.shp' → 匹配 HDR 中类别名称 '耕地'")
    print("  • 包含 Unclassified(0) 参与精度评价，结果与 ENVI 一致")
    print("=" * 80)

    # 交互输入
    reclass_dat = input("\n1. 重分类结果文件路径（_reclass.dat）: ").strip().strip('"\'')
    shp_folder = input("2. SHP 验证文件文件夹路径: ").strip().strip('"\'')
    out_dir = input("3. 输出目录: ").strip().strip('"\'')
    os.makedirs(out_dir, exist_ok=True)

    # ---- 1. 读取重分类结果影像 ----
    class_img, transform, crs = read_envi_classification(reclass_dat)

    # ---- 2. 从对应的 HDR 中读取类别名称 ----
    # 自动查找对应的 HDR 文件
    base = os.path.splitext(reclass_dat)[0]
    hdr_path = base + '.hdr'
    all_class_names = read_class_names_from_hdr(hdr_path)

    # 构建名称→代码映射（跳过 Unclassified(0) 和 Border(1)）
    # class names 索引 == 类别代码值（存储在 .dat 中）
    class_name_to_code = {
        name: idx
        for idx, name in enumerate(all_class_names)
        if idx >= 2  # 只取用户定义的类别
    }
    print(f"\n🔗 名称→代码映射:")
    for name, code in class_name_to_code.items():
        print(f"   '{name}' → 代码 {code}")

    # ---- 3. 读取 SHP 文件夹（按名称匹配） ----
    gdf, class_info = load_shp_folder(shp_folder, class_name_to_code)

    # ---- 3b. 注入"未分类"类别代码0，匹配ENVI精度评价方式 ----
    class_info.insert(0, (0, '未分类', 0))
    print(f"   含'未分类(0)'共 {len(class_info)} 个类别参与精度评价")

    # ---- 4. 坐标系匹配 ----
    if crs and gdf.crs and gdf.crs != crs:
        print(f"\n🔄 坐标系转换: {gdf.crs} -> {crs}")
        gdf = gdf.to_crs(crs)

    # ---- 5. 提取像元（保留未分类像元） ----
    real_codes, pred_codes = extract_pixels(gdf, class_img, transform,
                                            skip_nodata=False, shrink_pixels=0)
    if len(real_codes) == 0:
        print("❌ 无有效像素，请检查 SHP 是否与影像重叠")
        sys.exit(1)

    # ---- 6. 计算精度 ----
    cm, oa, kappa, producer, user, class_names = compute_metrics(real_codes, pred_codes, class_info)

    # ---- 7. 保存结果 ----
    base_name = os.path.splitext(os.path.basename(reclass_dat))[0]
    txt_out = os.path.join(out_dir, f"{base_name}_accuracy_report.txt")
    csv_out = os.path.join(out_dir, f"{base_name}_confusion_matrix.csv")
    export_report(cm, oa, kappa, producer, user, class_names, class_info, txt_out, csv_out)

    print("\n" + "=" * 80)
    print("🎉 精度评价完成！")
    print(f"   总体精度: {oa:.2f}%")
    print(f"   Kappa: {kappa:.4f}")
    print("=" * 80)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
