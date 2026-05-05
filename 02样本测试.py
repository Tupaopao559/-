# -*- coding: utf-8 -*-
"""
样本点数据处理流水线工具（整数序列保持版）
✅ 临时文件存放在D盘（程序运行完自动删除）
✅ 保持数据的整数格式，不转换为浮点数
✅ 每个单元格包含空格分隔的多个整数
✅ 按序列长度进行精确过滤
"""

import os
import sys
import numpy as np
import pandas as pd
import chardet
from glob import glob
from tqdm import tqdm
import math
import warnings
import gc
from collections import defaultdict
import tempfile
import shutil

warnings.filterwarnings('ignore')


def detect_encoding(file_path):
    """自动检测文件编码"""
    with open(file_path, 'rb') as f:
        raw_data = f.read(50000)
        result = chardet.detect(raw_data)
        return result['encoding'] if result['confidence'] > 0.7 else 'utf-8'


def count_tif_files(tif_folder):
    """统计TIF文件数量"""
    tif_files = sorted(f for f in os.listdir(tif_folder) if f.lower().endswith(('.tif', '.tiff')))
    return len(tif_files)


def find_first_valid_tif(tif_folder):
    """查找第一个有效的TIF文件"""
    tif_files = sorted(f for f in os.listdir(tif_folder) if f.lower().endswith(('.tif', '.tiff')))
    if not tif_files:
        return None
    return os.path.join(tif_folder, tif_files[0])


def lazy_import_geospatial():
    """延迟导入地理空间相关库"""
    try:
        import geopandas as gpd
        import rasterio
        from rasterio.transform import rowcol
        return gpd, rasterio, rowcol
    except ImportError as e:
        print(f"❌ 地理空间库导入失败: {e}")
        print("请确保已安装: pip install geopandas rasterio")
        sys.exit(1)


def create_temp_dir_on_d():
    """在D盘创建临时目录"""
    d_temp_base = "D:\\temp_sample_processing"
    os.makedirs(d_temp_base, exist_ok=True)

    # 创建唯一命名的临时子目录
    import time
    timestamp = int(time.time() * 1000) % 1000000  # 取毫秒后6位
    temp_dir = os.path.join(d_temp_base, f"temp_{timestamp}")
    os.makedirs(temp_dir, exist_ok=True)

    return temp_dir


def extract_points_with_pixel_location_batch(shp_input, raster_path, output_dir, batch_size=10000):
    """批量提取样本点对应影像像素值及行列位置"""
    gpd, rasterio, rowcol = lazy_import_geospatial()

    print("=== 🌍 提取样本点对应影像像素值及行列位置 ===")
    os.makedirs(output_dir, exist_ok=True)

    # 获取SHP文件列表
    if os.path.isfile(shp_input) and shp_input.lower().endswith('.shp'):
        shp_files = [shp_input]
    elif os.path.isdir(shp_input):
        shp_files = [os.path.join(shp_input, f) for f in os.listdir(shp_input)
                     if f.lower().endswith('.shp')]
    else:
        print(f"❌ 路径无效: {shp_input}")
        return False

    if not shp_files:
        print(f"⚠️ 未找到SHP文件: {shp_input}")
        return False

    # 加载影像数据
    try:
        with rasterio.open(raster_path) as src:
            transform = src.transform
            crs = src.crs
            band = src.read(1)
            nodata = src.nodata
            height, width = band.shape
    except Exception as e:
        print(f"❌ 无法打开影像文件: {e}")
        return False

    results_summary = []

    for shp_path in shp_files:
        class_name = os.path.splitext(os.path.basename(shp_path))[0]
        output_csv = os.path.join(output_dir, f"{class_name}.csv")

        print(f"\n📌 处理样本点: {os.path.basename(shp_path)}")

        try:
            # 分块读取SHP文件（针对大文件）
            gdf = gpd.read_file(shp_path)
            if len(gdf) == 0:
                print("   ⚠️ 该文件为空，跳过...")
                continue

            original_count = len(gdf)
            print(f"   📊 样本数量: {original_count:,}")

            # 坐标系匹配
            if gdf.crs != crs:
                gdf = gdf.to_crs(crs)

            # 批量提取坐标
            print("   🔄 批量提取像素值...")

            # 预分配数组
            pixel_values = [None] * original_count
            pixel_rows = [-1] * original_count
            pixel_cols = [-1] * original_count
            point_x = [0.0] * original_count
            point_y = [0.0] * original_count

            # 向量化处理
            geometries = gdf.geometry
            for i in tqdm(range(0, original_count, batch_size),
                          desc="提取进度",
                          total=math.ceil(original_count / batch_size)):
                end_idx = min(i + batch_size, original_count)
                batch_geoms = geometries.iloc[i:end_idx]

                # 批量计算行列号
                for j, geom in enumerate(batch_geoms):
                    idx = i + j
                    x, y = geom.x, geom.y
                    row, col = rowcol(transform, x, y)

                    point_x[idx] = x
                    point_y[idx] = y
                    pixel_rows[idx] = row
                    pixel_cols[idx] = col

                    if 0 <= row < height and 0 <= col < width:
                        val = band[row, col]
                        if not (np.isclose(val, nodata) if nodata is not None else np.isnan(val)):
                            pixel_values[idx] = int(val)  # 保持整数格式

            # 创建结果DataFrame
            result_df = pd.DataFrame({
                'pixel_value': pixel_values,
                'pixel_row': pixel_rows,
                'pixel_col': pixel_cols,
                'point_x': point_x,
                'point_y': point_y
            })

            # 添加原始属性（如果存在）
            for col in gdf.columns:
                if col != 'geometry':
                    result_df[col] = gdf[col].values

            # 保存结果
            result_df.to_csv(output_csv, index=False, encoding='utf-8')

            valid_count = sum(1 for v in pixel_values if v is not None)
            print(f"   ✅ 提取完成！有效值: {valid_count:,} / {original_count:,}")
            results_summary.append([class_name, original_count, valid_count, "success"])

            # 清理内存
            del gdf, result_df
            gc.collect()

        except Exception as e:
            print(f"   ❌ 处理失败: {e}")
            import traceback
            traceback.print_exc()
            results_summary.append([class_name, 0, 0, "error"])

    # 保存汇总
    if results_summary:
        summary_df = pd.DataFrame(results_summary,
                                  columns=['Class', 'Total_Points', 'Valid_Values', 'Status'])
        summary_csv = os.path.join(output_dir, "summary.csv")
        summary_df.to_csv(summary_csv, index=False, encoding='utf-8')
        print(f"\n📄 汇总报告已保存至: {summary_csv}")

    return True


def parse_cell_value_keep_integers(cell_value):
    """
    解析单元格值，保持整数格式
    如果包含空格分隔的多个数值则返回字符串（保持原格式），
    否则返回单一数值
    """
    if pd.isna(cell_value) or cell_value == '':
        return None

    cell_str = str(cell_value).strip()

    # 检查是否包含空格分隔的多个值
    if ' ' in cell_str:
        # 直接返回原始字符串（保持整数格式）
        return cell_str
    else:
        # 单一数值
        try:
            # 如果是单一数值，也转换为字符串格式
            return str(int(float(cell_str)))
        except ValueError:
            return None


def load_experiment_data_as_matrix(csv_path):
    """加载实验数据并解析多值序列（保持整数格式）"""
    print(f"📖 加载实验数据: {os.path.basename(csv_path)}")

    encoding = detect_encoding(csv_path)
    df = pd.read_csv(csv_path, header=None, dtype=str)  # 以字符串形式读取

    print(f"📊 实验数据维度: {df.shape[0]} 行 × {df.shape[1]} 列")

    # 解析每个单元格的值
    parsed_data = {}

    for i in tqdm(range(df.shape[0]), desc="解析数据", total=df.shape[0]):
        for j in range(df.shape[1]):
            cell_value = df.iloc[i, j]
            parsed_val = parse_cell_value_keep_integers(cell_value)
            parsed_data[(i, j)] = parsed_val

    print(f"✅ 解析完成，共处理 {len(parsed_data)} 个单元格")

    return parsed_data, df.shape


def extract_values_from_parsed_matrix(coords_df, parsed_data, matrix_shape, required_length):
    """从解析后的数据中提取值（保持字符串格式）"""
    total_points = len(coords_df)
    extracted = [None] * total_points

    rows = coords_df['pixel_row'].values
    cols = coords_df['pixel_col'].values

    for i in tqdm(range(total_points), desc="提取进度"):
        r = int(rows[i]) if not pd.isna(rows[i]) else -1
        c = int(cols[i]) if not pd.isna(cols[i]) else -1

        if 0 <= r < matrix_shape[0] and 0 <= c < matrix_shape[1]:
            # 获取解析后的值（保持字符串格式）
            parsed_val = parsed_data.get((r, c), None)
            if parsed_val is not None:
                # 检查序列长度
                if ' ' in parsed_val:
                    parts = parsed_val.split(' ')
                    if len(parts) == required_length:
                        extracted[i] = parsed_val
                else:
                    # 单一值也保持原格式
                    extracted[i] = parsed_val

    return extracted


def filter_by_sequence_length(df, sequence_column, required_length):
    """按序列长度过滤样本（保持字符串格式）"""

    if sequence_column not in df.columns:
        print(f"⚠️  列 '{sequence_column}' 不存在")
        return df.copy()

    def count_sequence_length(sequence_str):
        if pd.isna(sequence_str) or sequence_str == '':
            return 0
        parts = str(sequence_str).split(' ')
        parts = [part.strip() for part in parts if part.strip()]
        return len(parts)

    df['sequence_length'] = df[sequence_column].apply(count_sequence_length)
    filtered_df = df[df['sequence_length'] == required_length].copy()
    filtered_df = filtered_df.drop(columns=['sequence_length'])

    print(f"🔍 序列长度过滤: {len(df)} → {len(filtered_df)} (长度={required_length})")

    return filtered_df


def process_samples_optimized(sample_coords_path, experiment_data_path, output_dir, required_length):
    """优化处理样本（保持整数格式）"""

    base_name = os.path.splitext(os.path.basename(sample_coords_path))[0]
    filtered_output = os.path.join(output_dir, f"{base_name}.csv")

    try:
        # 1. 加载样本点
        print(f"\n📖 加载样本点文件: {os.path.basename(sample_coords_path)}")
        coords_df = pd.read_csv(sample_coords_path)
        print(f"✅ 加载 {len(coords_df):,} 个样本点")

        if len(coords_df) == 0:
            print("⚠️  样本点文件为空")
            # 创建空的输出文件
            empty_df = pd.DataFrame(columns=coords_df.columns.tolist())
            empty_df.to_csv(filtered_output, index=False, encoding='utf-8-sig')
            return True

        # 2. 加载实验数据
        print(f"📊 加载实验数据: {os.path.basename(experiment_data_path)}")
        parsed_data, matrix_shape = load_experiment_data_as_matrix(experiment_data_path)

        # 3. 提取值
        print("🔄 提取单元格值...")
        extracted_values = extract_values_from_parsed_matrix(coords_df, parsed_data, matrix_shape, required_length)
        coords_df['提取值'] = extracted_values

        # 4. 过滤
        print(f"\n🔍 按序列长度 {required_length} 过滤样本...")
        filtered_df = filter_by_sequence_length(coords_df, '提取值', required_length)

        # 5. 保存结果
        if len(filtered_df) > 0:
            filtered_df.to_csv(filtered_output, index=False, encoding='utf-8-sig')
            print(f"📋 保存 {len(filtered_df):,} 条记录到: {filtered_output}")
        else:
            # 创建空文件
            empty_df = pd.DataFrame(columns=coords_df.columns.tolist())
            empty_df.to_csv(filtered_output, index=False, encoding='utf-8-sig')
            print(f"📋 无符合条件样本，创建空文件: {filtered_output}")

        # 6. 内存清理
        del coords_df, filtered_df, parsed_data
        gc.collect()

        return True

    except Exception as e:
        print(f"❌ 处理失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main(shp_input, tif_folder, experiment_data_path, output_dir):
    """主处理流程"""

    print("=" * 70)
    print("样本点数据处理流水线工具（整数序列保持版）")
    print("=" * 70)

    # 自动检测序列长度
    required_length = count_tif_files(tif_folder)
    if required_length == 0:
        print("❌ 未找到TIF文件")
        return False

    print(f"📊 检测到 {required_length} 个TIF文件，将作为序列长度过滤条件")

    # 验证输入路径
    for path, desc in [(shp_input, "SHP路径"), (tif_folder, "TIF文件夹"),
                       (experiment_data_path, "实验数据CSV")]:
        if not os.path.exists(path):
            print(f"❌ {desc}不存在: {path}")
            return False

    # 创建临时目录在D盘
    temp_dir = create_temp_dir_on_d()
    print(f"📁 临时目录创建在: {temp_dir}")

    try:
        # Step 06: 提取样本点
        print("\n" + "=" * 70)
        print("步骤06: 提取样本点像素值和行列号")
        print("=" * 70)

        raster_path = find_first_valid_tif(tif_folder)
        if not raster_path:
            print("❌ 找不到有效的TIF文件")
            return False

        if not extract_points_with_pixel_location_batch(shp_input, raster_path, temp_dir):
            print("❌ 步骤06失败")
            return False

        # 获取生成的CSV文件
        step06_files = [f for f in os.listdir(temp_dir)
                        if f.endswith('.csv') and f != 'summary.csv']

        if not step06_files:
            print("⚠️  步骤06未生成CSV文件")
            return False

        # Step 08: 处理每个文件
        print("\n" + "=" * 70)
        print(f"步骤08: 处理 {len(step06_files)} 个样本文件")
        print("=" * 70)

        success_count = 0
        for csv_file in step06_files:
            csv_path = os.path.join(temp_dir, csv_file)
            print(f"\n📄 处理: {csv_file}")

            if process_samples_optimized(csv_path, experiment_data_path,
                                         output_dir, required_length):
                success_count += 1
                print(f"✅ {csv_file} 处理完成")
            else:
                print(f"❌ {csv_file} 处理失败")

        # 输出结果
        print("\n" + "=" * 70)
        print("处理完成!")
        print(f"✅ 成功处理: {success_count}/{len(step06_files)} 个文件")
        print(f"📁 输出目录: {output_dir}")
        print(f"🔍 过滤条件: 序列长度 = {required_length}")
        print("=" * 70)

        return success_count == len(step06_files)

    finally:
        # 清理D盘临时目录
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)
            print(f"🧹 清理D盘临时目录: {temp_dir}")


if __name__ == "__main__":

    print("=" * 70)
    print("样本点数据处理流水线工具（整数序列保持版）")
    print("=" * 70)
    print("功能特点:")
    print("  • 临时文件存放在D盘（程序运行完自动删除）")
    print("  • 保持数据的整数格式，不转换为浮点数")
    print("  • 每个单元格包含空格分隔的多个整数")
    print("  • 按序列长度进行精确过滤")
    print("  • 仅输出过滤后的筛选结果")
    print("=" * 70)

    # 获取输入
    inputs = [
        ("SHP文件/文件夹路径", "shp_input"),
        ("TIF影像文件夹路径", "tif_folder"),
        ("实验数据CSV文件路径", "experiment_data_path"),
        ("输出文件夹路径", "output_dir")
    ]

    params = {}
    for prompt, key in inputs:
        value = input(f"请输入 {prompt}: ").strip().strip('"\'')
        params[key] = value

        # 验证路径
        if key != 'output_dir' and not os.path.exists(value):
            print(f"❌ 路径不存在: {value}")
            sys.exit(1)

    # 创建输出目录
    os.makedirs(params['output_dir'], exist_ok=True)

    # 运行主程序
    if main(params['shp_input'], params['tif_folder'],
            params['experiment_data_path'], params['output_dir']):
        print("\n🎉 处理成功完成!")
    else:
        print("\n❌ 处理失败!")
        sys.exit(1)
