# make_boxes_reg.py
import os
from astropy.table import Table

# ============ 参数设置 ============
CATALOG_FILE = r"\\wsl$\Ubuntu\home\liqiang\datapool\CNEOST_test\star\test.cat"         # SExtractor 输出的 LDAC 星表
OUTPUT_REG   = r"\\wsl$\Ubuntu\home\liqiang\datapool\CNEOST_test\star\seg_fits\all_boxes.reg"     # 生成的区域文件名
BOX_WIDTH    = 23                  # 框的宽度（像素）
BOX_HEIGHT   = 23                  # 框的高度（像素）
# 如果希望框的大小与 VIGNET 一致，就设成 VIGNET(40,40) 的尺寸
# =================================

def create_boxes_reg():
    # 1. 读取星表
    print(f"正在读取星表: {CATALOG_FILE}")
    cat = Table.read(CATALOG_FILE, hdu=2)   # LDAC 星表在第 2 个 HDU

    # 检查必须的坐标列
    if "XWIN_IMAGE" not in cat.colnames or "YWIN_IMAGE" not in cat.colnames:
        raise KeyError("星表中缺少 XWIN_IMAGE / YWIN_IMAGE 列，请检查 .param 文件。")

    n_sources = len(cat)
    print(f"共找到 {n_sources} 个源，开始生成 .reg 文件...")

    # 2. 生成区域文件内容
    lines = []
    # 文件头（DS9 格式版本声明，颜色等）
    lines.append("# Region file format: DS9 version 4.1")
    lines.append("global color=red width=2 font=\"helvetica 10 normal roman\" select=1 highlite=1")
    lines.append("physical")   # 使用物理像素坐标（从1开始）

    # 3. 遍历每个源，画矩形框
    n = 0
    for source in cat:
        x = source["XWIN_IMAGE"]    # 单位：像素，坐标原点在左下角 (1,1)
        y = source["YWIN_IMAGE"]
        # 矩形 box 需要左下角坐标和宽、高
        # DS9 的 box 语法: box(x, y, width, height, angle=0)
        # 我们以一个角为参考，通常以源中心为基准计算左下角
        x1 = x
        y1 = y
        lines.append(f"box({x1:.4f},{y1:.4f},{BOX_WIDTH},{BOX_HEIGHT},0)")
        n += 1

    # 4. 写入文件
    with open(OUTPUT_REG, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"✅ 已生成 {OUTPUT_REG}，包含 {n} 个矩形框。")

if __name__ == "__main__":
    create_boxes_reg()