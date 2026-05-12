# save_vignettes.py
import os
import numpy as np
from astropy.io import fits
from astropy.table import Table

# ============== 参数设置 ==============
CATALOG_FILE = r"\\wsl$\Ubuntu\home\liqiang\datapool\CNEOST_test\star\test.cat"          # SExtractor 输出星表（FITS_LDAC 格式）
# OUTPUT_DIR   = r"\\wsl$\Ubuntu\home\liqiang\datapool\CNEOST_test\star\seg_fits" # 输出文件夹
OUTPUT_DIR   = r"D:\Pycharm_proj\PythonProject\data\normal"
VIGNET_SIZE  = (23, 23)           # 和 .param 里 VIGNET(n,m) 一致
# =====================================

def save_vignettes():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # 1. 读取星表 (LDAC 格式星表通常在第 2 个 HDU)
    print(f"正在读取星表: {CATALOG_FILE}")
    try:
        cat = Table.read(CATALOG_FILE, hdu=2)
    except Exception as e:
        raise RuntimeError(f"读取 LDAC 星表失败，请检查文件格式: {e}")

    # 检查是否包含 VIGNET 列
    if "VIGNET" not in cat.colnames:
        raise KeyError("星表中没有 'VIGNET' 列，请在 .param 文件中加上 VIGNET(30,30) 再重新运行 SExtractor。")

    n_sources = len(cat)
    print(f"星表中共有 {n_sources} 个源，开始保存...")

    # 2. 遍历每个源
    for i, source in enumerate(cat):
        obj_id = source["NUMBER"]                  # 编号
        vignet_1d = source["VIGNET"]               # 一维数组

        # 重塑为二维图像
        try:
            vignet_2d = vignet_1d.reshape(VIGNET_SIZE)
        except ValueError:
            print(f"⚠️ 源 {obj_id}: VIGNET 尺寸不匹配，期望 {VIGNET_SIZE}，"
                  f"实际数据长度 {len(vignet_1d)}，跳过。")
            continue

        # 3. 写入 FITS
        mask = vignet_2d < 0.1  # 识别出极端负值的无效像素
        vignet_2d[mask] = 0
        hdu = fits.PrimaryHDU(vignet_2d.astype(np.float64))
        hdu.header['VIG_SIZE'] = f"{VIGNET_SIZE[0]}x{VIGNET_SIZE[1]}"
        hdu.header['NOTE'] = "Extracted from SExtractor VIGNET (background subtracted)"

        out_name = os.path.join(OUTPUT_DIR, f"vignet_{obj_id:05d}.fits")
        hdu.writeto(out_name, overwrite=True)

        if (i + 1) % 100 == 0:
            print(f"已处理 {i+1}/{n_sources} 个源...")

    print(f"🎉 完成！共保存 {n_sources} 个 VIGNET 至 {OUTPUT_DIR}")

if __name__ == "__main__":
    save_vignettes()