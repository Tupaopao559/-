import os
import numpy as np
import rasterio
from tqdm import tqdm
import re
import colorsys


def read_envi_header(hdr_path):
    """从 ENVI .hdr 文件中读取所有参数（包括地理信息）"""
    # 检查文件是否存在
    if not os.path.exists(hdr_path):
        # 尝试添加 .hdr 扩展名
        if not hdr_path.endswith('.hdr'):
            hdr_path_with_ext = hdr_path + '.hdr'
            if os.path.exists(hdr_path_with_ext):
                print(f"⚠️  自动修复路径: {hdr_path} -> {hdr_path_with_ext}")
                hdr_path = hdr_path_with_ext
            else:
                raise FileNotFoundError(f"找不到ENVI头文件: {hdr_path} 或 {hdr_path + '.hdr'}")

    # 检查是否是文件
    if not os.path.isfile(hdr_path):
        raise PermissionError(f"路径不是文件，可能是文件夹: {hdr_path}")

    # 检查文件权限
    if not os.access(hdr_path, os.R_OK):
        raise PermissionError(f"无权限读取文件: {hdr_path}")

    print(f"📖 正在读取头文件: {hdr_path}")

    # 尝试 GBK 优先（ENVI 中文版常用），失败则回退 UTF-8
    content = None
    for encoding in ['gbk', 'utf-8']:
        try:
            with open(hdr_path, 'r', encoding=encoding) as f:
                content = f.read()
            if content.strip():
                break
        except UnicodeDecodeError:
            continue
    if not content:
        raise ValueError(f"无法读取 HDR 文件（尝试了 GBK 和 UTF-8）: {hdr_path}")

    # 提取关键参数
    samples = int(re.search(r'samples\s*=\s*(\d+)', content, re.IGNORECASE).group(1))
    lines = int(re.search(r'lines\s*=\s*(\d+)', content, re.IGNORECASE).group(1))
    bands = int(re.search(r'bands\s*=\s*(\d+)', content, re.IGNORECASE).group(1))

    # 判断数据类型
    data_type_map = {
        '1': np.uint8,
        '2': np.int16,
        '3': np.int32,
        '4': np.float32,
        '5': np.float64,
        '6': np.complex64,
        '9': np.complex128,
        '12': np.uint16,
        '13': np.uint32,
        '14': np.int64,
        '15': np.uint64,
    }
    data_type_code = re.search(r'data type\s*=\s*(\d+)', content, re.IGNORECASE)
    dtype = data_type_map.get(data_type_code.group(1), np.int16) if data_type_code else np.int16

    # 提取地理信息（如果存在）
    geographic_info = {}
    geographic_patterns = [
        (r'map info\s*=\s*\{([^}]*)\}', 'map_info'),
        (r'coordinate system string\s*=\s*\{([^}]*)\}', 'coordinate_system'),
        (r'projection info\s*=\s*\{([^}]*)\}', 'projection_info'),
        (r'x start\s*=\s*([^\n]+)', 'x_start'),
        (r'y start\s*=\s*([^\n]+)', 'y_start'),
        (r'pixel size\s*=\s*\{([^\}]*)\}', 'pixel_size'),
        (r'wavelength units\s*=\s*([^\n]+)', 'wavelength_units'),
        (r'band names\s*=\s*\{([^}]*)\}', 'band_names'),
        (r'data ignore value\s*=\s*([^\n]+)', 'data_ignore_value'),
        (r'sensor type\s*=\s*([^\n]+)', 'sensor_type'),
        (r'description\s*=\s*\{([^}]*)\}', 'description')
    ]

    for pattern, key in geographic_patterns:
        match = re.search(pattern, content, re.IGNORECASE)
        if match:
            geographic_info[key] = match.group(1).strip()

    return samples, lines, bands, dtype, geographic_info


def reclassify_with_geospatial_info(dat_path, class_value_groups, class_names):
    """
    对 ENVI 格式的 .dat 文件进行重分类，并保留地理坐标信息
    class_value_groups: 每个类别的数值列表，如[[1,2,7], [3,4,8], [5,6,9]]
    class_names: 类别名称列表，如["耕地", "林地", "水体"]
    """
    # 1. 检查并修复 dat_path
    if not os.path.exists(dat_path):
        # 尝试添加 .dat 扩展名
        if not dat_path.endswith('.dat'):
            possible_dat = dat_path + '.dat'
            if os.path.exists(possible_dat):
                print(f"⚠️  自动修复DAT路径: {dat_path} -> {possible_dat}")
                dat_path = possible_dat
            else:
                print(f"❌ 错误：找不到DAT文件: {dat_path}")
                print(f"   尝试的路径: {possible_dat}")
                return False

    print(f"📂 处理文件: {dat_path}")

    # 2. 构建 hdr_path
    if dat_path.endswith('.dat'):
        hdr_path = dat_path[:-4] + '.hdr'  # 替换扩展名
    else:
        hdr_path = dat_path + '.hdr'  # 直接添加扩展名

    print(f"📂 对应头文件: {hdr_path}")

    # 3. 检查头文件是否存在
    if not os.path.exists(hdr_path):
        print(f"❌ 错误：找不到对应的HDR文件: {hdr_path}")
        print("💡 请确保以下文件存在:")
        print(f"   1. DAT文件: {dat_path}")
        print(f"   2. HDR文件: {hdr_path}")
        return False

    # 4. 读取原始 .hdr 获取尺寸、数据类型和地理信息
    try:
        samples, lines, bands, dtype, geo_info = read_envi_header(hdr_path)
    except Exception as e:
        print(f"❌ 读取HDR文件失败: {e}")
        return False

    print(f"📊 图像尺寸: {lines} 行 × {samples} 列 × {bands} 波段, 数据类型: {dtype}")
    print(f"📍 地理信息: {len(geo_info)} 项")
    if geo_info:
        for key, value in list(geo_info.items())[:3]:  # 只显示前3项
            print(f"   {key}: {value[:50]}..." if len(str(value)) > 50 else f"   {key}: {value}")

    # 5. 读取 .dat 数据
    print("📥 读取数据文件...")
    try:
        data = np.fromfile(dat_path, dtype=dtype)
        print(f"✅ 读取成功，数据大小: {data.shape}, 总元素: {data.size}")
    except Exception as e:
        print(f"❌ 读取DAT文件失败: {e}")
        return False

    # 6. 重塑数据
    total_elements = lines * samples
    if bands > 1:
        total_elements_with_bands = total_elements * bands
        if data.size != total_elements_with_bands:
            print(f"⚠️  警告：数据大小不匹配!")
            print(f"   期望: {total_elements_with_bands} 元素 ({bands} 波段)")
            print(f"   实际: {data.size} 元素")
            if data.size == total_elements:
                print("   自动调整为单波段数据")
                bands = 1
            else:
                # ⭐ 修复：报告尺寸不匹配但试图从 HDR 推断正确尺寸
                # 通过 data.size 反推实际可能的总元素数（取最接近的整除数）
                actual_total = int(np.sqrt(data.size)) ** 2  # 尝试取平方
                for candidate_h in range(int(np.sqrt(data.size)), lines * 2):
                    if data.size % candidate_h == 0:
                        candidate_w = data.size // candidate_h
                        if abs(candidate_h * candidate_w - data.size) < 0.001:
                            print(f"   自动检测到实际尺寸: {candidate_h} × {candidate_w}")
                            lines = candidate_h
                            samples = candidate_w
                            total_elements = lines * samples
                            break
                else:
                    print(f"❌ 无法自动匹配数据尺寸，请检查 HDR 与实际数据是否一致")
                    return False

        data = data.reshape((bands, lines, samples))
        if bands > 1:
            data = data[0]  # 取第一波段
            print(f"✅ 使用第1波段进行分类（共{bands}波段）")
    else:
        if data.size != total_elements:
            print(f"⚠️  警告：数据大小不匹配!")
            print(f"   期望: {total_elements} 元素 ({samples}×{lines} = {total_elements})")
            print(f"   实际: {data.size} 元素")
            # ⭐ 修复：尝试自动检测正确尺寸，而非直接截断
            # 算出行数和列数最接近的整数因子
            if data.size > total_elements:
                # 可能 HDR 里 lines/samples 被写错了，尝试用 data.size 反推
                for candidate_h in range(lines, int(np.sqrt(data.size)) * 2):
                    if data.size % candidate_h == 0:
                        candidate_w = data.size // candidate_h
                        print(f"   自动检测到实际尺寸: {candidate_h} × {candidate_w} (原 HDR: {lines}×{samples})")
                        lines = candidate_h
                        samples = candidate_w
                        total_elements = lines * samples
                        break
                else:
                    print(f"❌ 无法自动匹配数据尺寸")
                    print(f"   可能原因：ENVI 文件包含额外的文件头或元数据")
                    print(f"   建议：用 rasterio 而非 np.fromfile 读取该文件")
                    return False
            else:
                print(f"❌ 数据比预期少，无法处理")
                return False

        data = data.reshape((lines, samples))

    # 7. 重分类 - 根据用户定义的数值组进行重分类
    print("🔄 进行重分类...")
    reclassified = np.zeros_like(data, dtype=np.uint8)  # 分类结果用 uint8 足够

    # 统计原始数据分布
    unique_values = np.unique(data)
    print(f"📈 原始数据唯一值: {unique_values[:20]}{'...' if len(unique_values) > 20 else ''}")

    # 特殊值处理：-1值标记为边框（类别1），0值保持未分类（类别0）
    border_mask = data == -1
    reclassified[border_mask] = 1  # 边框类别
    border_count = np.sum(border_mask)
    print(f"   边框(-1值): {border_count} 个像素")

    # 根据用户定义的数值组进行重分类（从类别2开始，因为0=未分类，1=边框）
    for i, value_group in enumerate(class_value_groups):
        # 对于当前类别，将所有属于该组的数值都标记为类别编号i+2（跳过0和1）
        for value in value_group:
            # 排除-1和0这两个特殊值
            if value == -1 or value == 0:
                print(f"   跳过特殊值 {value} (已在特殊处理中)")
                continue

            mask = (data == value) & (reclassified == 0)  # 只处理还没有被分类的像素
            count = np.sum(mask)
            if count > 0:
                reclassified[mask] = i + 2  # 类别从2开始
                print(f"   类别{i + 2}({class_names[i]}): 值{value} -> {count}个像素")

    # 统计未分类像素（排除边框-1值，只统计0值）
    unclassified_count = np.sum((reclassified == 0) & (data != -1))  # 只统计非-1的0值
    border_count_final = np.sum(reclassified == 1)  # 边框数量
    total_pixels = lines * samples
    classified_count = total_pixels - unclassified_count - border_count_final
    print(f"📊 分类统计:")
    print(f"   总像素: {total_pixels}")
    print(f"   已分类: {classified_count} ({classified_count / total_pixels * 100:.2f}%)")
    print(f"   边框(-1值): {border_count_final} ({border_count_final / total_pixels * 100:.2f}%)")
    print(f"   未分类(0值): {unclassified_count} ({unclassified_count / total_pixels * 100:.2f}%)")

    # 8. 保存重分类后的 .dat
    base_name = os.path.splitext(dat_path)[0]
    if base_name.endswith('.dat'):
        base_name = base_name[:-4]
    new_dat_path = base_name + '_reclass.dat'

    print(f"💾 保存重分类数据: {new_dat_path}")
    try:
        reclassified.tofile(new_dat_path)
        print(f"✅ 重分类数据已保存: {new_dat_path}")
    except Exception as e:
        print(f"❌ 保存DAT文件失败: {e}")
        return False

    # 9. 创建带地理信息的分类 .hdr 文件
    print("📝 创建头文件...")

    # 生成颜色列表（包括特殊类别：未分类=灰色，边框=黑色）
    # 0=未分类(灰色)，1=边框(黑色)，然后是用户定义的类别
    colors = [
        '128, 128, 128',  # 未分类 - 灰色
        '0, 0, 0',  # 边框 - 黑色
        '0, 255, 0',  # 类别1 - 绿色
        '0, 0, 255',  # 类别2 - 蓝色
        '255, 0, 0',  # 类别3 - 红色
        '34, 139, 34',  # 类别4 - 森林绿
        '255, 255, 0',  # 类别5 - 黄色
        '255, 0, 255',  # 类别6 - 洋红
        '0, 255, 255',  # 类别7 - 青色
        '128, 0, 128',  # 类别8 - 紫色
        '255, 165, 0',  # 类别9 - 橙色
        '128, 128, 0',  # 类别10 - 橄榄色
        '139, 69, 19',  # 类别11 - 巧克力色
        '255, 192, 203',  # 类别12 - 粉色
        '165, 42, 42',  # 类别13 - 棕色
        '0, 128, 0',  # 类别14 - 深绿色
        '0, 0, 128',  # 类别15 - 深蓝色
        '128, 0, 0',  # 类别16 - 深红色
        '255, 140, 0',  # 类别17 - 深橙色
        '75, 0, 130',  # 类别18 - 靛蓝色
        '255, 20, 147',  # 类别19 - 深粉色
        '0, 255, 127',  # 类别20 - 春绿色
        '100, 149, 237',  # 类别21 - 钢蓝色
        '218, 165, 32',  # 类别22 - 金色
        '255, 182, 193',  # 类别23 - 桃色
        '210, 105, 30',  # 类别24 - 番茄色
        '255, 69, 0',  # 类别25 - 橙红色
        '184, 134, 11',  # 类别26 - 沙褐色
        '0, 255, 255',  # 类别27 - 深青色
        '138, 43, 226',  # 类别28 - 蓝紫色
        '255, 105, 180',  # 类别29 - 热粉色
        '255, 218, 185'  # 类别30 - 蜜桃色
    ]

    # 如果类别数超过预定义颜色数，动态生成颜色
    if len(class_value_groups) + 2 > len(colors):  # +2 for unclassified and border
        for i in range(len(colors), len(class_value_groups) + 2):
            hue = i / (len(class_value_groups) + 2)
            rgb = tuple(int(c * 255) for c in colorsys.hsv_to_rgb(hue, 1.0, 1.0))
            colors.append(f'{rgb[0]}, {rgb[1]}, {rgb[2]}')

    # 计算总类别数（包括未分类类和边框类）
    total_classes = len(class_value_groups) + 2  # +2 for unclassified(0) and border(1)

    # 构建HDR文件内容
    header_content = f"""ENVI
description = {{重分类结果 - {os.path.basename(dat_path)}}}
samples = {samples}
lines = {lines}
bands = 1
header offset = 0
file type = ENVI Classification
data type = 1
interleave = bsq
byte order = 0
classes = {total_classes}
class names = {{Unclassified, Border, """

    # 添加类别名称
    for i, name in enumerate(class_names):
        if i < len(class_names) - 1:
            header_content += f"{name}, "
        else:
            header_content += f"{name}}}\n"

    # 添加class lookup
    header_content += "class lookup = {"
    color_values = []
    for i in range(total_classes):
        if i < len(colors):
            color_values.append(colors[i])
        else:
            # 如果类别超过预定义颜色数量，使用默认颜色
            color_values.append(f"{i * 30 % 255}, {i * 50 % 255}, {i * 70 % 255}")

    header_content += ", ".join(color_values) + "}\n"

    # 添加原始地理信息
    if geo_info:
        print("📍 保留地理信息...")
        for key, value in geo_info.items():
            if key == 'map_info':
                # 确保map info格式正确
                header_content += f"map info = {{{value}}}\n"
            elif key == 'coordinate_system':
                header_content += f"coordinate system string = {{{value}}}\n"
            elif key == 'projection_info':
                header_content += f"projection info = {{{value}}}\n"
            elif key == 'x start':
                header_content += f"x start = {value}\n"
            elif key == 'y start':
                header_content += f"y start = {value}\n"
            elif key == 'pixel_size':
                header_content += f"pixel size = {{{value}}}\n"
            elif key == 'wavelength_units':
                header_content += f"wavelength units = {value}\n"
            elif key == 'band_names':
                header_content += f"band names = {{{value}}}\n"
            elif key == 'data_ignore_value':
                header_content += f"data ignore value = {value}\n"
            elif key == 'sensor_type':
                header_content += f"sensor type = {value}\n"
            # description已跳过

    # 写入HDR文件
    new_hdr_path = new_dat_path.replace('.dat', '.hdr')
    try:
        # 使用 GBK 编码保存，ENVI 中文版用 ANSI/GBK 读取 .hdr
        # 如果编辑器打开乱码，请切换到 GBK/ANSI 编码查看
        with open(new_hdr_path, 'w', encoding='gbk') as f:
            f.write(header_content)
        print(f"✅ 重分类头文件已保存: {new_hdr_path} (编码: GBK)")
    except UnicodeEncodeError as e:
        # 保底：如果GBK编码失败，用GBK的replace模式
        print(f"⚠️  部分字符无法用GBK编码: {e}")
        print("   使用GBK保底模式保存（无法编码的字符替换为?）")
        try:
            with open(new_hdr_path, 'w', encoding='gbk', errors='replace') as f:
                f.write(header_content)
        except Exception as e2:
            print(f"❌ 保存HDR文件失败: {e2}")
            return False
    except Exception as e:
        print(f"❌ 保存HDR文件失败: {e}")
        return False

    print("=" * 60)
    print("🎉 重分类完成！")
    print("=" * 60)
    print(f"   输出数据文件: {new_dat_path}")
    print(f"   输出头文件:   {new_hdr_path}")
    print(f"   图像尺寸:     {lines} × {samples}")
    print(f"   分类类别数:   {total_classes}")
    print(f"   已分类像素:   {classified_count / total_pixels * 100:.2f}%")
    print(f"   边框像素:     {border_count_final / total_pixels * 100:.2f}%")
    print(f"   未分类像素:   {unclassified_count / total_pixels * 100:.2f}%")
    print(f"   地理信息:     {len(geo_info)}项")

    return True


def main():
    print("=" * 80)
    print("   ENVI格式数据重分类工具（增强版-支持特殊值处理）")
    print("=" * 80)
    print("   特殊值处理：")
    print("   • 0值 -> 未分类类（灰色）")
    print("   • -1值 -> 边框类（黑色）")
    print("   • 其他值 -> 用户定义的分类")
    print("=" * 80)

    # 获取用户输入
    print("\n📂 请输入ENVI数据文件路径：")
    print("   提示：可以输入DAT文件路径，或没有扩展名的文件名")
    dat_path = input("文件路径: ").strip().strip('"\'')

    # 检查路径是否存在
    if not os.path.exists(dat_path):
        print(f"⚠️  警告：路径不存在: {dat_path}")
        print("   正在尝试自动修复...")

        # 尝试常见扩展名
        possible_paths = [
            dat_path + '.dat',
            dat_path,
            dat_path.replace('\\', '/'),
        ]

        for path in possible_paths:
            if os.path.exists(path):
                dat_path = path
                print(f"✅ 找到文件: {dat_path}")
                break
        else:
            print(f"❌ 错误：找不到文件: {dat_path}")
            print("💡 请检查:")
            print("   1. 文件是否存在")
            print("   2. 路径是否正确")
            print("   3. 是否有文件扩展名（如.dat）")
            return False

    print(f"✅ 确认文件: {dat_path}")

    print("\n📊 请输入重分类参数：")
    try:
        num_classes = int(input("请输入地物类别数量: ").strip())
        if num_classes <= 0:
            print("❌ 错误：类别数量必须大于0")
            return False
    except ValueError:
        print("❌ 错误：请输入有效的数字")
        return False

    class_value_groups = []
    class_names = []

    for i in range(num_classes):
        print(f"\n--- 第 {i + 1}/{num_classes} 类地物设置 ---")
        name = input(f"请输入第 {i + 1} 类的名称: ").strip()
        if not name:
            print("⚠️  警告：类别名称不能为空，使用默认名称")
            name = f"Class_{i + 1}"

        values_input = input(f"请输入第 {i + 1} 类包含的数值（用逗号分隔，如：1,2,7）: ").strip()
        if not values_input:
            print("❌ 错误：数值不能为空")
            return False

        try:
            values = [int(x.strip()) for x in values_input.split(',')]
            class_value_groups.append(values)
            class_names.append(name)
            print(f"✅ 类别 {i + 2}: '{name}' -> 数值 {values}")  # 从2开始编号（0=未分类，1=边框）
        except ValueError:
            print("❌ 错误：请输入有效的数字")
            return False

    print(f"\n📋 分类配置总结:")
    print(f"   总类别数: {len(class_names)} (不含特殊类别)")
    print(f"   特殊类别:")
    print(f"     0 -> 未分类 (灰色) - 原始数据中的0值")
    print(f"     1 -> 边框 (黑色) - 原始数据中的-1值")
    for i, (values, name) in enumerate(zip(class_value_groups, class_names)):
        print(f"   类别 {i + 2}: '{name}' -> 数值 {values}")

    confirm = input("\n⚠️  确认开始重分类？(y/n): ").strip().lower()
    if confirm != 'y':
        print("🚫 用户取消操作")
        return False

    # 执行重分类
    print("\n" + "=" * 60)
    print("🔄 开始重分类处理...")
    print("=" * 60)

    if not reclassify_with_geospatial_info(dat_path, class_value_groups, class_names):
        print("❌ 重分类处理失败")
        return False

    print("\n" + "=" * 80)
    print("✅ 处理流程完成！")
    output_dir = os.path.dirname(dat_path) or "."
    print(f"📁 结果文件已保存至: {output_dir}")

    # 显示生成的文件
    base_name = os.path.splitext(dat_path)[0]
    if base_name.endswith('.dat'):
        base_name = base_name[:-4]

    reclass_dat = base_name + '_reclass.dat'
    reclass_hdr = base_name + '_reclass.hdr'

    if os.path.exists(reclass_dat):
        file_size = os.path.getsize(reclass_dat) / (1024 * 1024)  # MB
        print(f"📊 生成文件:")
        print(f"   {reclass_dat} ({file_size:.2f} MB)")
        print(f"   {reclass_hdr}")

    print("=" * 80)

    return True


if __name__ == "__main__":
    try:
        if main():
            print("\n🎉 重分类处理完成！")
            print("💡 提示：可以在ENVI软件中打开 _reclass.dat 文件查看分类结果")
            print("   特殊值颜色：")
            print("   • 0值 (未分类) - 灰色")
            print("   • -1值 (边框) - 黑色")
            print("   • 其他值 - 用户定义的颜色")
        else:
            print("\n❌ 重分类处理失败！")
    except KeyboardInterrupt:
        print("\n\n⚠️ 用户中断操作。")
    except Exception as e:
        print(f"\n❌ 程序异常: {e}")
        import traceback

        traceback.print_exc()