# -*- coding: utf-8 -*-
"""
ENVI 分类精度评价（基于面状 ROI）
- 强制 NoData = 1，有效类别 ≥2
- ROI 中必须包含一个字段，存储分类图对应的像素值（如 2,3,4...）
- 自动使用该字段进行对应，无需手动输入映射
- 输出混淆矩阵、OA、Kappa、用户/生产者精度
"""

import os
import sys
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio import features
from sklearn.metrics import confusion_matrix, cohen_kappa_score, accuracy_score
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


def read_envi_classification(dat_path, forced_nodata=1):
    """读取 ENVI 分类结果，强制 NoData = 1"""
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
    print(f"   NoData 强制设置为: {forced_nodata}")
    unique_vals = np.unique(data)
    print(f"   影像唯一值: {unique_vals}")
    return data, forced_nodata, transform, crs


def load_roi_with_pixel_value(shp_path):
    """
    加载 ROI Shapefile，自动识别或要求用户指定存储像素值的字段。
    返回 GeoDataFrame，并添加 'true_code' 列（整数，即分类图对应的值）
    """
    if not os.path.exists(shp_path):
        raise FileNotFoundError(f"Shapefile 不存在: {shp_path}")
    gdf = gpd.read_file(shp_path)
    print(f"\n📁 ROI 文件: {os.path.basename(shp_path)}")
    print(f"   多边形数量: {len(gdf)}")
    print(f"   字段列表: {list(gdf.columns)}")

    # 自动查找可能的像素值字段（常见名称）
    candidates = ['Value', 'value', 'code', 'class_value', 'class_code', 'pixel_value', 'class']
    found_field = None
    for fld in candidates:
        if fld in gdf.columns:
            found_field = fld
            break
    if found_field:
        print(f"   🔍 自动识别像素值字段: '{found_field}'")
        use_field = found_field
    else:
        use_field = input("请输入存储分类图像素值的字段名: ").strip()
        if use_field not in gdf.columns:
            raise ValueError(f"字段 '{use_field}' 不存在")

    # 提取像素值，转为整数
    gdf['true_code'] = pd.to_numeric(gdf[use_field], errors='coerce')
    # 删除无效值或小于2的值（NoData=1）
    gdf = gdf[gdf['true_code'].notna() & (gdf['true_code'] >= 2)]
    gdf = gdf[gdf.geometry.is_valid & ~gdf.geometry.is_empty]
    print(f"   有效多边形: {len(gdf)} (已排除无像素值或<2的记录)")
    if len(gdf) == 0:
        raise ValueError("没有有效多边形，请检查像素值字段是否正确且值≥2")
    codes = sorted(gdf['true_code'].unique())
    print(f"   包含的像素值: {codes}")
    return gdf


def extract_pixels(gdf, class_img, transform, nodata):
    """
    提取每个多边形内部所有像元的真实代码和预测代码。
    返回: (real_codes, pred_codes)
    """
    h, w = class_img.shape
    real_codes = []
    pred_codes = []
    class_img_int = class_img.astype(np.int32)

    for idx, row in tqdm(gdf.iterrows(), total=len(gdf), desc="处理ROI多边形"):
        geom = row.geometry
        true_code = int(row['true_code'])

        try:
            mask = features.geometry_mask([geom], out_shape=(h, w), transform=transform, invert=True)
        except Exception as e:
            print(f"   警告: 多边形 {idx} 掩膜失败: {e}")
            continue

        pixels = class_img_int[mask]
        # 排除 NoData (=1) 和小于2的值（背景）
        valid_mask = (pixels != nodata) & (pixels >= 2)
        valid_pixels = pixels[valid_mask]
        if len(valid_pixels) == 0:
            continue

        real_codes.extend([true_code] * len(valid_pixels))
        pred_codes.extend(valid_pixels.tolist())

    print(f"\n✅ 共提取 {len(real_codes)} 个有效像元")
    return real_codes, pred_codes


def compute_metrics(real_codes, pred_codes):
    """
    使用原始像素值计算混淆矩阵和精度。
    自动找出所有出现的真实类别代码，按代码值排序作为行列顺序。
    """
    unique_codes = sorted(set(real_codes))
    code_to_idx = {code: i for i, code in enumerate(unique_codes)}
    idx_to_code = {i: code for i, code in enumerate(unique_codes)}
    class_names = [str(code) for code in unique_codes]

    # 转换真实标签
    real_idx = [code_to_idx[c] for c in real_codes]
    # 预测标签：只保留也在 unique_codes 中的，否则丢弃（避免混淆矩阵形状不匹配）
    pred_idx = []
    for p in pred_codes:
        if p in code_to_idx:
            pred_idx.append(code_to_idx[p])
        else:
            # 预测值不在真实类别中，记录但跳过（不放入混淆矩阵，因为没有对应行）
            pass
    # 同步过滤真实标签
    filtered_real = []
    filtered_pred = []
    for r, p in zip(real_codes, pred_codes):
        if p in code_to_idx:
            filtered_real.append(code_to_idx[r])
            filtered_pred.append(code_to_idx[p])

    n_skipped = len(real_codes) - len(filtered_real)
    if n_skipped > 0:
        print(f"⚠️ 过滤掉 {n_skipped} 个预测值不在真实类别中的像素（分类图可能存在额外类别）")

    n_classes = len(unique_codes)
    labels = list(range(n_classes))
    cm = confusion_matrix(filtered_real, filtered_pred, labels=labels)
    oa = accuracy_score(filtered_real, filtered_pred) * 100
    kappa = cohen_kappa_score(filtered_real, filtered_pred)

    producer = []
    user = []
    for i in range(n_classes):
        total_actual = cm[i, :].sum()
        total_pred = cm[:, i].sum()
        producer.append(cm[i, i] / total_actual * 100 if total_actual > 0 else 0)
        user.append(cm[i, i] / total_pred * 100 if total_pred > 0 else 0)

    return cm, oa, kappa, producer, user, class_names, idx_to_code


def export_report(cm, oa, kappa, producer, user, class_names, idx_to_code, output_txt, output_csv):
    with open(output_txt, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("ENVI 分类精度评价报告（基于 ROI 像素值字段自动对应）\n")
        f.write(f"NoData = 1，有效类别 = {class_names}\n")
        f.write("=" * 80 + "\n\n")
        f.write("混淆矩阵（行列以像素值标识）:\n")
        f.write("真实\\预测\t" + "\t".join(class_names) + "\n")
        for i, name in enumerate(class_names):
            f.write(f"{name}\t" + "\t".join(map(str, cm[i, :])) + "\n")
        f.write(f"\n总体精度 (OA): {oa:.2f}%\n")
        f.write(f"Kappa 系数: {kappa:.4f}\n\n")
        f.write("各类别精度:\n")
        f.write(f"{'类别(像素值)':<15} {'生产者精度 (%)':<20} {'用户精度 (%)':<20}\n")
        for i, name in enumerate(class_names):
            f.write(f"{name:<15} {producer[i]:<20.2f} {user[i]:<20.2f}\n")
        f.write(f"{'平均':<15} {np.mean(producer):<20.2f} {np.mean(user):<20.2f}\n")

    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    cm_df.to_csv(output_csv, encoding='utf-8-sig')
    print(f"\n📄 报告已保存: {output_txt}")
    print(f"📊 混淆矩阵 CSV: {output_csv}")


def main():
    print("=" * 80)
    print("ENVI 分类精度评价 - 自动使用 ROI 中的像素值字段")
    print("=" * 80)

    dat_path = input("\n1. ENVI 分类结果文件路径（.dat）: ").strip().strip('"\'')
    shp_path = input("2. ROI Shapefile 路径（需包含像素值字段）: ").strip().strip('"\'')
    out_dir = input("3. 输出目录: ").strip().strip('"\'')
    os.makedirs(out_dir, exist_ok=True)

    # 读取分类图
    class_img, nodata, transform, crs = read_envi_classification(dat_path, forced_nodata=1)

    # 读取 ROI（自动识别像素值字段）
    gdf = load_roi_with_pixel_value(shp_path)

    # 坐标系匹配
    if crs and gdf.crs and gdf.crs != crs:
        print(f"\n🔄 坐标系转换: {gdf.crs} -> {crs}")
        gdf = gdf.to_crs(crs)

    # 提取像素
    real_codes, pred_codes = extract_pixels(gdf, class_img, transform, nodata)
    if len(real_codes) == 0:
        print("❌ 无有效像素，请检查 ROI 是否与影像重叠，以及像素值字段是否正确")
        sys.exit(1)

    # 计算精度
    cm, oa, kappa, producer, user, class_names, idx_to_code = compute_metrics(real_codes, pred_codes)

    # 保存结果
    base = os.path.splitext(os.path.basename(dat_path))[0]
    txt_out = os.path.join(out_dir, f"{base}_accuracy_report.txt")
    csv_out = os.path.join(out_dir, f"{base}_confusion_matrix.csv")
    export_report(cm, oa, kappa, producer, user, class_names, idx_to_code, txt_out, csv_out)

    print("\n🎉 精度评价完成！")
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