import os
import csv
import pandas as pd
import numpy as np
import rasterio
from tqdm import tqdm
from glob import glob
import sys


def get_image_count_from_folder(folder_path):
    """从文件夹中统计影像图数量"""
    supported_extensions = ['.tif', '.tiff', '.img']
    image_files = []

    for ext in supported_extensions:
        image_files.extend(glob(os.path.join(folder_path, f'*{ext}')))
        image_files.extend(glob(os.path.join(folder_path, f'*{ext.upper()}')))

    # 去重
    image_files = sorted(set(image_files))
    return len(image_files), image_files


def process_csv_pandas(input_file, output_file, target_length, max_sequence_length=None):
    """
    Pandas库处理：将指定长度数字空格间隔的单元格改为0，空单元格也赋值为0
    同时，将序列长度大于max_sequence_length的单元格也改为0
    """
    print(f"开始处理CSV文件: {os.path.basename(input_file)}")
    print(f"目标序列长度: {target_length} (基于影像图数量)")
    if max_sequence_length:
        print(f"最大序列长度: {max_sequence_length} (用户输入)")

    try:
        # 检查文件是否存在
        if not os.path.exists(input_file):
            print(f"❌ 错误：文件不存在 → {input_file}")
            return False

        # 读取CSV（使用header=None避免自动解析列名）
        df = pd.read_csv(input_file, dtype=str, header=None, keep_default_na=False)
        print(f"读取成功：{len(df)} 行，{len(df.columns)} 列")

        # 定义处理函数：判断单元格是否符合条件并替换，空单元格直接赋值为0，保留-1值
        def replace_target_cell(cell):
            if cell.strip() == "":  # 空单元格判断
                return '0'
            # 保留-1值
            if cell.strip() == "-1":
                return '-1'
            # 处理包含空格分隔数字的单元格
            cell_parts = cell.strip().split()
            # 检查是否为有效的数字序列
            try:
                valid_parts = []
                for part in cell_parts:
                    if part.strip():
                        float(part)
                        valid_parts.append(part)
                # 检查序列长度是否等于目标长度或大于最大序列长度
                if len(valid_parts) == target_length or (
                        max_sequence_length and len(valid_parts) > max_sequence_length):
                    # 将符合条件的值转换为0
                    return '0'
            except:
                pass
            # 保持其他值不变
            return cell

        # 批量应用处理函数到所有单元格
        print("正在处理数据...")

        df_processed = df.apply(lambda row: row.apply(replace_target_cell), axis=1)

        # 确保输出目录存在
        output_dir = os.path.dirname(output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        # 保存结果（不写入索引，保持原始结构）
        try:
            df_processed.to_csv(output_file, index=False, header=False, encoding='utf-8')
            print(f"CSV处理完成！结果已保存至：{output_file}")
            return True
        except PermissionError as e:
            print(f"权限错误：无法写入文件 → {output_file}")
            print(f"错误详情：{str(e)}")
            return False
        except Exception as e:
            print(f"保存文件出错：{str(e)}")
            return False

    except ImportError:
        print("❌ 未检测到Pandas库，请先执行安装：pip install pandas")
        return False
    except Exception as e:
        print(f"❌ CSV处理出错：{str(e)}")
        import traceback
        traceback.print_exc()
        return False


def csv_to_txt(input_csv, output_txt):
    """
    将CSV文件转换为TXT格式
    """
    print(f"开始转换CSV到TXT: {os.path.basename(input_csv)}")

    try:
        # 检查文件是否存在
        if not os.path.exists(input_csv):
            print(f"❌ 错误：文件不存在 → {input_csv}")
            return False

        # 读取CSV
        try:
            with open(input_csv, 'r', encoding='utf-8') as csvfile:
                reader = csv.reader(csvfile)
                data = list(reader)
            print(f"读取成功：{len(data)} 行数据")
        except UnicodeDecodeError:
            print("UTF-8编码读取失败，尝试使用GBK编码...")
            with open(input_csv, 'r', encoding='gbk') as csvfile:
                reader = csv.reader(csvfile)
                data = list(reader)
            print(f"读取成功：{len(data)} 行数据")

        # 确保输出目录存在
        output_dir = os.path.dirname(output_txt)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        # 写入TXT
        try:
            with open(output_txt, 'w', encoding='utf-8') as txtfile:
                for row_idx, row in enumerate(tqdm(data, desc="转换进度", unit="行")):
                    # ⭐ 关键修复：每个 CSV 单元格必须精确对应 1 个 TXT 值
                    # 如果单元格有空格（未匹配的序列），只取第一个值
                    # 确保 TXT 行列数 = CSV 行列数 = 原始影像行列数
                    cleaned_row = []
                    for cell in row:
                        cell = cell.strip()
                        if not cell:
                            cleaned_row.append('0')
                        elif ' ' in cell:
                            # 序列未匹配，取第一个值（通常是分类值或0）
                            first_val = cell.split()[0]
                            cleaned_row.append(first_val)
                        else:
                            cleaned_row.append(cell)
                    txtfile.write(' '.join(cleaned_row) + '\n')

            print(f"CSV转TXT完成！结果已保存至：{output_txt}")
            return True
        except PermissionError as e:
            print(f"权限错误：无法写入文件 → {output_txt}")
            print(f"错误详情：{str(e)}")
            return False
        except Exception as e:
            print(f"写入文件出错：{str(e)}")
            return False

    except Exception as e:
        print(f"❌ CSV转TXT出错：{str(e)}")
        import traceback
        traceback.print_exc()
        return False


def txt_to_envi_with_metadata(txt_file_path, output_file_path, ref_tif_path=None):
    """
    将TXT文件转换为ENVI格式，并可选地添加地理坐标信息
    修改：输出与参考TIF相同尺寸，但只在有效数据位置填充值，其余为NoData
    """
    print(f"开始转换TXT到ENVI: {os.path.basename(txt_file_path)}")
    if ref_tif_path:
        print(f"   参考TIF文件: {os.path.basename(ref_tif_path)}")

    try:
        # 检查文件是否存在
        if not os.path.exists(txt_file_path):
            print(f"❌ 错误：TXT文件不存在 → {txt_file_path}")
            return False

        # 读取参考TIF文件获取尺寸信息
        if ref_tif_path and os.path.exists(ref_tif_path):
            print("从参考TIF文件获取地理信息和尺寸...")
            try:
                with rasterio.open(ref_tif_path) as src_ref:
                    ref_width = src_ref.width
                    ref_height = src_ref.height
                    transform = src_ref.transform
                    crs = src_ref.crs
                    print(f"   参考影像尺寸: {ref_width} × {ref_height}")
                    print(f"   坐标系统: {crs}")
            except Exception as e:
                print(f"无法读取参考TIF文件: {e}")
                return False
        else:
            print("未提供参考TIF文件，使用默认尺寸")
            # 使用默认尺寸
            ref_width = 1000
            ref_height = 1000
            transform = None
            crs = None
            print(f"   默认影像尺寸: {ref_width} × {ref_height}")

        # 读取TXT文件
        try:
            with open(txt_file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except UnicodeDecodeError:
            print("⚠️ UTF-8编码读取失败，尝试使用GBK编码...")
            with open(txt_file_path, 'r', encoding='gbk') as f:
                lines = f.readlines()

        # 解析每一行数据
        txt_rows = []
        for line in lines:
            line = line.strip()
            if line:
                # 将字符串转换为数字列表
                numeric_vals = []
                for val in line.split():
                    try:
                        num_val = float(val)
                        numeric_vals.append(num_val)
                    except ValueError:
                        numeric_vals.append(0.0)
                if numeric_vals:
                    txt_rows.append(numeric_vals)

        if not txt_rows:
            print("TXT文件中没有找到有效的数字数据")
            return False

        # ⭐ 修复：用 TXT 的实际尺寸，代替参考 TIF 的尺寸
        txt_actual_rows = len(txt_rows)
        txt_actual_cols = max(len(row) for row in txt_rows) if txt_rows else 0
        print(f"TXT数据实际尺寸: {txt_actual_rows} 行 × {txt_actual_cols} 列")

        # 验证尺寸是否合理（与参考 TIF 对比，超出时告警但不截断）
        if ref_tif_path and os.path.exists(ref_tif_path):
            if txt_actual_rows != ref_height or txt_actual_cols != ref_width:
                print(f"⚠️  警告：TXT 尺寸 ({txt_actual_rows}×{txt_actual_cols}) "
                      f"与参考 TIF ({ref_height}×{ref_width}) 不匹配！")
                print(f"   将使用 TXT 实际尺寸输出，忽略参考 TIF 尺寸")
                emit_geo_anyway = True
            else:
                emit_geo_anyway = False
        else:
            emit_geo_anyway = False

        # ⭐ 用 TXT 的实际尺寸创建矩阵，彻底杜绝截断填充导致的条纹
        output_height = txt_actual_rows
        output_width = txt_actual_cols
        data = np.full((output_height, output_width), -9999.0, dtype=np.float32)
        print(f"创建输出矩阵尺寸: {output_height} × {output_width} (基于 TXT 实际数据)")

        # 将TXT数据按行填入输出矩阵
        for i in range(output_height):
            row_data = txt_rows[i]
            actual_len = len(row_data)
            if actual_len >= output_width:
                data[i, :] = row_data[:output_width]
            else:
                data[i, :actual_len] = row_data
                # 剩余部分保持 -9999 (NoData)，而非 0，避免将 NoData 误设为 0 类
                data[i, actual_len:] = -9999.0

        # 确保输出目录存在
        output_dir = os.path.dirname(output_file_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        # 保存为ENVI格式
        print("正在保存ENVI文件...")
        try:
            with rasterio.open(
                    output_file_path,
                    'w',
                    driver='ENVI',
                    height=output_height,
                    width=output_width,
                    count=1,
                    dtype=data.dtype,
                    crs=crs if not emit_geo_anyway else None,  # 尺寸不同时不写地理信息
                    transform=transform if not emit_geo_anyway else None,
                    nodata=-9999  # 设置NoData值
            ) as dst:
                dst.write(data, 1)
                dst.update_tags(ENVI={'description': 'Use Rainbow color table for better visualization'})
        except PermissionError as e:
            print(f"权限错误：无法写入ENVI文件 → {output_file_path}")
            print(f"错误详情：{str(e)}")
            return False
        except Exception as e:
            print(f"保存ENVI文件出错：{str(e)}")
            return False

        # 创建HDR文件
        hdr_file = output_file_path + '.hdr'
        try:
            with open(hdr_file, 'w', encoding='utf-8') as f:
                f.write('ENVI\n')
                f.write(f'description = {{Use Rainbow color table for better visualization}}\n')
                f.write(f'samples = {output_width}\n')  # 列数
                f.write(f'lines = {output_height}\n')  # 行数
                f.write(f'bands = 1\n')
                f.write('header offset = 0\n')
                f.write('file type = ENVI Standard\n')
                f.write('data type = 4\n')  # 4对应float32
                f.write('interleave = bsq\n')
                f.write('byte order = 0\n')

                # 地理信息（仅当尺寸匹配时写入，否则 NoData 区域的地理参考无意义）
                if crs and transform and not emit_geo_anyway:
                    x_start = transform[2]
                    y_start = transform[5]
                    pixel_width = transform[0]
                    pixel_height = abs(transform[4])

                    f.write(f'x start = {x_start}\n')
                    f.write(f'y start = {y_start}\n')
                    f.write(
                        f'map info = {{UTM, 1, 1, {x_start}, {y_start}, {pixel_width}, {pixel_height}, North, 0}}\n')
                    f.write(f'coordinate system string = {crs}\n')
                    f.write('projection info = {UTM, North, 0, 0, 0, 0, 0, 0}\n')

                f.write('wavelength units = Unknown\n')
                f.write('band names = {Band 1}\n')
                f.write('data ignore value = -9999\n')  # NoData值
                f.write('sensor type = Unknown\n')
        except PermissionError as e:
            print(f"权限错误：无法写入HDR文件 → {hdr_file}")
            print(f"错误详情：{str(e)}")
            return False
        except Exception as e:
            print(f"保存HDR文件出错：{str(e)}")
            return False

        print(f"ENVI文件已生成: {output_file_path}")
        print(f"HDR文件已生成: {hdr_file}")

        # 统计信息
        valid_data_count = np.count_nonzero(data != -9999)
        total_count = data.size
        print(f"有效数据占比: {valid_data_count / total_count * 100:.2f}% ({valid_data_count}/{total_count} 像素)")

        return True

    except Exception as e:
        print(f"❌ TXT转ENVI失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def create_temp_dir():
    """创建临时目录"""
    temp_base = os.path.join(os.getcwd(), "temp_envi_processing")
    os.makedirs(temp_base, exist_ok=True)

    # 创建唯一命名的临时子目录
    import time
    timestamp = int(time.time() * 1000) % 1000000  # 取毫秒后6位
    temp_dir = os.path.join(temp_base, f"temp_{timestamp}")
    os.makedirs(temp_dir, exist_ok=True)

    return temp_dir


def cleanup_temp_dir(temp_dir):
    """清理临时目录"""
    try:
        if os.path.exists(temp_dir):
            import shutil
            shutil.rmtree(temp_dir)
            print(f"临时目录已清理: {temp_dir}")
    except Exception as e:
        print(f"清理临时目录失败: {e}")


def main(input_csv, images_folder, output_dir, add_geo=True, max_sequence_length=None):
    print("=" * 70)
    print("   CSV → TXT → ENVI 自动化处理链（动态序列长度，支持地理坐标）")
    print("=" * 70)
    print("此工具将自动完成以下流程：")
    print("1. CSV处理：将指定长度数字序列单元格替换为0（基于影像图数量确定长度）")
    print("2. CSV转TXT：将CSV格式转换为空格分隔的TXT")
    print("3. TXT转ENVI：将TXT转换为ENVI格式栅格文件（可选带地理坐标）")
    print("=" * 70)

    # 验证输入路径
    if not os.path.exists(input_csv):
        print(f"错误：输入文件不存在！ → {input_csv}")
        return False

    if not os.path.exists(images_folder):
        print(f"错误：影像图文件夹不存在！ → {images_folder}")
        return False

    # 创建临时目录
    temp_dir = create_temp_dir()
    print(f"临时目录创建在: {temp_dir}")

    # 验证输出目录权限
    use_temp_dir = False
    try:
        # 检查输出目录的写入权限
        test_file = os.path.join(output_dir, "test_write_permission.txt")
        with open(test_file, 'w') as f:
            f.write("test")
        os.remove(test_file)
        print(f"输出目录: {output_dir}")
        print("输出目录写入权限正常")
    except Exception as e:
        print(f"警告：输出目录可能没有写入权限: {e}")
        print("使用临时目录作为替代输出目录...")
        use_temp_dir = True
        output_dir = temp_dir
        print(f"输出目录: {output_dir}")

    # 获取影像图数量
    image_count, image_files = get_image_count_from_folder(images_folder)
    if image_count == 0:
        print("在影像图文件夹中未找到任何支持的影像文件")
        print("支持的格式: .tif, .tiff, .img")
        return False

    print(f"\n检测到 {image_count} 个影像文件，将使用 {image_count} 作为目标序列长度")
    print("影像文件列表:")
    for i, img_file in enumerate(image_files[:5]):  # 只显示前5个
        print(f"  {i + 1}: {os.path.basename(img_file)}")
    if len(image_files) > 5:
        print(f"  ... 还有 {len(image_files) - 5} 个文件")

    # 确定参考TIF文件
    ref_tif = None
    if add_geo:
        # 从影像文件夹中自动选择第一张TIF文件
        tif_files = [f for f in image_files if f.lower().endswith(('.tif', '.tiff'))]
        if tif_files:
            ref_tif = tif_files[0]
            print(f"自动选择参考TIF文件: {os.path.basename(ref_tif)}")
        else:
            print("在影像文件夹中未找到TIF文件，无法添加地理坐标")
            add_geo = False

    # 生成输出文件路径
    base_name = os.path.splitext(os.path.basename(input_csv))[0]

    processed_csv = os.path.join(output_dir, f"{base_name}_processed.csv")
    output_txt = os.path.join(output_dir, f"{base_name}_converted.txt")
    output_envi = os.path.join(output_dir, f"{base_name}_result.dat")

    print(f"\n处理流程：")
    print(f"   输入CSV: {input_csv}")
    print(f"   影像文件夹: {images_folder} ({image_count} 张影像)")
    if ref_tif:
        print(f"   参考TIF: {ref_tif}")
    print(f"   中间CSV: {processed_csv}")
    print(f"   中间TXT: {output_txt}")
    print(f"   输出ENVI: {output_envi}")

    # 执行处理流程
    print("\n开始自动化处理...")

    # 步骤1: CSV处理（使用动态长度）
    if not process_csv_pandas(input_csv, processed_csv, image_count, max_sequence_length):
        print("第一步（CSV处理）失败，停止处理。")
        return False

    # 步骤2: CSV转TXT
    if not csv_to_txt(processed_csv, output_txt):
        print("第二步（CSV转TXT）失败，停止处理。")
        return False

    # 步骤3: TXT转ENVI（带地理坐标）
    if not txt_to_envi_with_metadata(output_txt, output_envi, ref_tif):
        print("第三步（TXT转ENVI）失败，停止处理。")
        return False

    print("\n" + "=" * 70)
    print("处理完成！所有步骤均已成功。")
    print(f"处理后的CSV文件: {processed_csv}")
    print(f"转换后的TXT文件: {output_txt}")
    print(f"最终ENVI文件: {output_envi}")
    print(f"ENVI头文件: {output_envi}.hdr")
    print("=" * 70)

    # 保留临时目录，不自动清理
    if use_temp_dir:
        print(f"临时目录已保留: {temp_dir}")
        print("您可以在该目录中找到所有处理结果文件")

    return True


if __name__ == "__main__":
    print("=" * 70)
    print("   CSV → TXT → ENVI 自动化处理链（动态序列长度，支持地理坐标）")
    print("=" * 70)
    print("功能特点:")
    print("  - 支持命令行参数，方便自动化运行")
    print("  - 基于影像图数量确定目标序列长度")
    print("  - 自动处理CSV、TXT转换和ENVI生成")
    print("  - 支持添加地理坐标信息")
    print("  - 完善的错误处理和权限检查")
    print("=" * 70)

    # 用户键盘输入
    print("请输入以下信息:")
    input_csv = input("1. 输入CSV文件路径: ")
    images_folder = input("2. 影像图文件夹路径: ")
    output_dir = input("3. 输出目录路径: ")
    add_geo_input = input("4. 是否添加地理坐标 (true/false，默认true): ")
    add_geo = add_geo_input.lower() == 'true' if add_geo_input else True

    # 用户输入序列长度
    sequence_length = input("5. 请输入序列长度（本次调试输入2）: ")
    try:
        sequence_length = int(sequence_length)
    except:
        sequence_length = 2  # 默认值
    print(f"使用序列长度: {sequence_length}")

    print(f"使用输入参数:")
    print(f"  输入CSV文件: {input_csv}")
    print(f"  影像图文件夹: {images_folder}")
    print(f"  输出目录: {output_dir}")
    print(f"  添加地理坐标: {add_geo}")
    print("=" * 70)

    # 验证路径
    for path, desc in [(input_csv, "输入CSV文件"), (images_folder, "影像图文件夹")]:
        if not os.path.exists(path):
            print(f"{desc}不存在: {path}")
            sys.exit(1)

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 运行主程序
    try:
        if main(input_csv, images_folder, output_dir, add_geo, sequence_length):
            print("\n处理成功完成!")
        else:
            print("\n处理失败!")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n用户中断操作。")
    except Exception as e:
        print(f"\n程序异常: {e}")
        import traceback

        traceback.print_exc()
