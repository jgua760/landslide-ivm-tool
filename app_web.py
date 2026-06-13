# -*- coding: utf-8 -*-
"""
滑坡易发性分析工具 - Web版 (Streamlit)
直接复用原 app.py 中的核心服务类，界面改为 Streamlit
"""

import streamlit as st
import os
import math
import tempfile
import zipfile
import shutil
import numpy as np
import pandas as pd
import traceback

st.set_page_config(
    page_title="滑坡易发性分析工具",
    page_icon="🏔️",
    layout="wide"
)

# ── GIS 依赖检查 ──────────────────────────────────────────────
try:
    import rasterio
    from rasterio.warp import reproject, Resampling
    from rasterio.features import geometry_mask
    import geopandas as gpd
    import mapclassify
    from shapely.geometry import Point
    from scipy.ndimage import binary_erosion
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    GIS_OK = True
except ImportError as _e:
    GIS_OK = False
    GIS_ERR = str(_e)


# ══════════════════════════════════════════════════════════════
# 核心服务类（直接复用原 app.py 逻辑，零修改）
# ══════════════════════════════════════════════════════════════

class DataClassificationService:
    def __init__(self, log_callback=None):
        self.log = log_callback if log_callback else print

    def clean_data(self, data, nodata_val):
        data = data.astype(np.float32)
        if nodata_val is not None:
            data[data == nodata_val] = np.nan
        for val in [32767, -32768, -9999, 9999, 65535, -2147483648, 2147483647]:
            data[np.isclose(data, val)] = np.nan
        data[data < -100000] = np.nan
        return data

    def classify_raster(self, data, nodata_val, factor_name):
        data = self.clean_data(data, nodata_val)
        fname_lower = factor_name.lower()
        valid_mask = ~np.isnan(data)
        valid_pixels = data[valid_mask]
        if valid_pixels.size == 0:
            return data, "无有效数据"
        unique_vals = np.unique(valid_pixels)
        if len(unique_vals) < 50:
            if np.all(np.isclose(unique_vals, np.round(unique_vals))):
                return data.astype(np.int32), "已分类数据(保持原值)"
        if 'slope' in fname_lower or '坡度' in fname_lower:
            data[(data < 0) | (data > 90)] = np.nan
            return self._classify_slope(data), "坡度每10度分级(自动)"
        if 'aspect' in fname_lower or '坡向' in fname_lower:
            data[(data < -1) | (data > 360)] = np.nan
            return self._classify_aspect(data), "坡向9方向重分类(自动)"
        return self._classify_jenks(data, valid_mask, valid_pixels, k=5), "自然断点法(k=5)"

    def _classify_slope(self, data):
        out_data = np.full(data.shape, -9999, dtype=np.int32)
        mask = ~np.isnan(data)
        if not np.any(mask):
            return out_data
        out_data[mask] = np.digitize(data[mask], [10, 20, 30, 40, 50]) + 1
        return out_data

    def _classify_aspect(self, data):
        out_data = np.full(data.shape, -9999, dtype=np.int32)
        mask = ~np.isnan(data)
        if not np.any(mask):
            return out_data
        d = data
        out_temp = np.zeros_like(d, dtype=np.int32)
        out_temp[d < 0] = 1
        out_temp[(d >= 0)    & (d < 22.5)]  = 2
        out_temp[(d >= 22.5) & (d < 67.5)]  = 3
        out_temp[(d >= 67.5) & (d < 112.5)] = 4
        out_temp[(d >= 112.5)& (d < 157.5)] = 5
        out_temp[(d >= 157.5)& (d < 202.5)] = 6
        out_temp[(d >= 202.5)& (d < 247.5)] = 7
        out_temp[(d >= 247.5)& (d < 292.5)] = 8
        out_temp[(d >= 292.5)& (d < 337.5)] = 9
        out_temp[d >= 337.5] = 2
        out_data[mask] = out_temp[mask]
        return out_data

    def _classify_jenks(self, data, mask, valid_pixels, k=5):
        try:
            sample = valid_pixels if valid_pixels.size <= 200000 else \
                     np.random.choice(valid_pixels, 200000, replace=False)
            classifier = mapclassify.NaturalBreaks(sample, k=k)
            bins = classifier.bins
            out_data = np.full(data.shape, -9999, dtype=np.int32)
            classified = np.digitize(data[mask], bins, right=True) + 1
            classified[classified > k] = k
            out_data[mask] = classified
            return out_data
        except Exception:
            return data


class GISAnalysisService:
    def __init__(self, log_callback=None):
        self.log = log_callback if log_callback else print
        self.classifier = DataClassificationService(log_callback)

    def _calculate_iv(self, ni, si, N, S):
        if N == 0 or S == 0:
            return 0.0
        lr = ni / N
        ar = si / S
        return math.log(lr / ar) if ar > 0 and lr > 0 else -5.0

    def classify_susceptibility_5class(self, accumulated_iv, global_valid_mask):
        valid_iv = accumulated_iv[global_valid_mask]
        valid_iv = valid_iv[np.isfinite(valid_iv)]
        if valid_iv.size == 0:
            raise ValueError("综合易发性图中没有有效像元，无法分级。")
        if np.unique(valid_iv).size < 5:
            raise ValueError("有效 IV 唯一值不足 5 个，无法执行 k=5 自然断点分级。")
        classifier = mapclassify.NaturalBreaks(valid_iv, k=5)
        bins = classifier.bins
        classified_iv = np.full(accumulated_iv.shape, -9999, dtype=np.int16)
        fvm = global_valid_mask & np.isfinite(accumulated_iv)
        classes = np.clip(np.digitize(accumulated_iv[fvm], bins, right=True) + 1, 1, 5)
        classified_iv[fvm] = classes.astype(np.int16)
        return classified_iv, bins

    def process_single_raster(self, raster_path, gdf_landslide, output_dir, reference_meta=None):
        factor_name = os.path.splitext(os.path.basename(raster_path))[0]
        self.log(f"处理: {factor_name}...")
        with rasterio.open(raster_path) as src:
            src_nodata = src.nodata
            if reference_meta and (src.width != reference_meta['width'] or
                                   src.height != reference_meta['height']):
                self.log("  -> 几何对齐重采样...")
                raw_data = np.empty(
                    (reference_meta['height'], reference_meta['width']), dtype=np.float32)
                reproject(
                    source=rasterio.band(src, 1), destination=raw_data,
                    src_transform=src.transform, src_crs=src.crs,
                    dst_transform=reference_meta['transform'],
                    dst_crs=reference_meta['crs'],
                    resampling=Resampling.nearest)
                transform = reference_meta['transform']
                profile = reference_meta.copy()
            else:
                raw_data = src.read(1)
                transform = src.transform
                profile = src.profile.copy()

        classified_data, method = self.classifier.classify_raster(
            raw_data, src_nodata, factor_name)
        classified_data = classified_data.astype(np.int32)
        processing_nodata = -9999
        current_mask = (classified_data != processing_nodata)
        valid_pixels = classified_data[current_mask]

        if valid_pixels.size == 0:
            self.log("  -> 警告: 缺少有效数据，跳过")
            return None, None, None, None

        class_counts = pd.Series(valid_pixels).value_counts().to_dict()
        total_pixels_S = valid_pixels.size
        total_landslides_N = len(gdf_landslide)
        coords = [(x, y) for x, y in zip(
            gdf_landslide.geometry.x, gdf_landslide.geometry.y)]
        rows, cols = rasterio.transform.rowcol(
            transform, [p[0] for p in coords], [p[1] for p in coords])

        landslide_in_classes = {}
        for r, c in zip(rows, cols):
            if 0 <= r < profile['height'] and 0 <= c < profile['width']:
                if current_mask[r, c]:
                    val = classified_data[r, c]
                    landslide_in_classes[val] = landslide_in_classes.get(val, 0) + 1

        iv_map, stats_list = {}, []
        for cls_val in sorted(class_counts):
            if cls_val == processing_nodata or cls_val < -9999 or cls_val > 999999:
                continue
            si = class_counts.get(cls_val, 0)
            ni = landslide_in_classes.get(cls_val, 0)
            iv = self._calculate_iv(ni, si, total_landslides_N, total_pixels_S)
            iv_map[cls_val] = iv
            stats_list.append({
                "因子": factor_name, "等级": cls_val,
                "Si": si, "Ni": ni, "IV": round(iv, 4), "方法": method
            })

        iv_raster_data = np.full(classified_data.shape, processing_nodata, dtype=np.float32)
        for cls_val, iv_val in iv_map.items():
            iv_raster_data[classified_data == cls_val] = iv_val

        profile.update(dtype=rasterio.float32, count=1, nodata=processing_nodata)
        with rasterio.open(
                os.path.join(output_dir, f"IV_{factor_name}.tif"), 'w', **profile) as dst:
            dst.write(iv_raster_data, 1)

        return pd.DataFrame(stats_list), iv_raster_data, current_mask, profile

    def run_batch_analysis(self, raster_dir, shp_path, output_dir, progress_callback):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        self.log(">>> [1/4] 读取正样本数据...")
        gdf = gpd.read_file(shp_path)
        tif_files = [f for f in os.listdir(raster_dir) if f.lower().endswith('.tif')]
        self.log(f">>> [2/4] 处理 {len(tif_files)} 个因子...")

        accumulated_iv = global_valid_mask = reference_profile = None
        all_stats = []

        for idx, tif in enumerate(tif_files):
            stats_df, raster_data, current_mask, current_profile = \
                self.process_single_raster(
                    os.path.join(raster_dir, tif), gdf, output_dir, reference_profile)
            if reference_profile is None and current_profile is not None:
                reference_profile = current_profile
            if stats_df is not None:
                all_stats.append(stats_df)
                if global_valid_mask is None:
                    global_valid_mask = current_mask.copy()
                    accumulated_iv = np.where(raster_data != -9999.0, raster_data, 0)
                else:
                    global_valid_mask &= current_mask
                    accumulated_iv += np.where(raster_data != -9999.0, raster_data, 0.0)
            progress_callback((idx + 1) / len(tif_files))

        self.log(">>> [3/4] 导出叠加结果...")
        if accumulated_iv is not None and reference_profile is not None:
            final_data = accumulated_iv.copy()
            final_data[~global_valid_mask] = -9999.0
            reference_profile.update(
                dtype=rasterio.float32, count=1, nodata=-9999.0, compress='lzw')
            final_path = os.path.join(output_dir, "Final_Susceptibility_Map.tif")
            with rasterio.open(final_path, 'w', **reference_profile) as dst:
                dst.write(final_data, 1)
            self.log(f"结果保存: {final_path}")

            self.log(">>> [4/4] 自然断点法(k=5)分级...")
            try:
                classified_iv, bins = self.classify_susceptibility_5class(
                    accumulated_iv, global_valid_mask)
                class_profile = reference_profile.copy()
                class_profile.update(
                    dtype=rasterio.int16, count=1, nodata=-9999, compress='lzw')
                class_path = os.path.join(
                    output_dir, "Classified_Susceptibility_Map_5Class.tif")
                with rasterio.open(class_path, 'w', **class_profile) as dst:
                    dst.write(classified_iv, 1)
                self.log(f"✅ 分级图层保存: {class_path}")
                self.log(f"   -> 5级阈值: {np.round(bins, 4)}")
            except Exception as e:
                self.log(f"❌ 分级失败: {e}")

        if all_stats:
            try:
                final_df = pd.concat(all_stats, ignore_index=True)
                csv_path = os.path.join(output_dir, "IV_Statistics.csv")
                final_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
                self.log(f"统计表保存: {csv_path}")
            except Exception as e:
                self.log(f"CSV失败: {e}")

        self.log("✅ IVM计算全部完成！")


class NegativeSampleService:
    def __init__(self, log_callback=None):
        self.log = log_callback if log_callback else print

    def generate_negative_samples(self, factor_dir, ivm_tif_path, pos_shp_path,
                                  out_shp_path, sample_mode="ivm_low", buffer_m=500):
        self.log(">>> [1/5] 读取滑坡正样本...")
        gdf_pos = gpd.read_file(pos_shp_path)
        target_n = len(gdf_pos)
        self.log(f"  -> 需生成负样本: {target_n} 个")

        self.log(">>> [2/5] 建立全局有效掩膜...")
        tif_files = [f for f in os.listdir(factor_dir) if f.lower().endswith('.tif')]
        if not tif_files:
            raise ValueError("环境因子目录中未找到TIF文件！")

        with rasterio.open(os.path.join(factor_dir, tif_files[0])) as src_ref:
            transform, crs, shape = src_ref.transform, src_ref.crs, src_ref.shape

        global_valid_mask = np.ones(shape, dtype=bool)
        for fname in tif_files:
            with rasterio.open(os.path.join(factor_dir, fname)) as src_f:
                if src_f.shape != shape or src_f.transform != transform:
                    data_f = np.empty(shape, dtype=np.float32)
                    reproject(
                        source=rasterio.band(src_f, 1), destination=data_f,
                        src_transform=src_f.transform, src_crs=src_f.crs,
                        dst_transform=transform, dst_crs=crs,
                        resampling=Resampling.nearest)
                    nodata_f = src_f.nodata if src_f.nodata is not None else -9999.0
                else:
                    data_f = src_f.read(1)
                    nodata_f = src_f.nodata if src_f.nodata is not None else -9999.0
                factor_valid = (~np.isnan(data_f)) & (data_f > -100000.0)
                if nodata_f is not None:
                    factor_valid &= ~np.isclose(data_f, nodata_f)
                global_valid_mask &= factor_valid

        self.log(">>> [3/5] 建立采样空间约束...")
        safe_area_mask = np.ones(shape, dtype=bool)
        if sample_mode in ["ivm_very_low", "ivm_low"]:
            if not ivm_tif_path or not os.path.exists(ivm_tif_path):
                raise ValueError("IVM约束模式需要提供有效的IVM TIF路径！")
            with rasterio.open(ivm_tif_path) as src_ivm:
                if src_ivm.shape != shape or src_ivm.transform != transform:
                    data_ivm = np.empty(shape, dtype=np.float32)
                    reproject(
                        source=rasterio.band(src_ivm, 1), destination=data_ivm,
                        src_transform=src_ivm.transform, src_crs=src_ivm.crs,
                        dst_transform=transform, dst_crs=crs,
                        resampling=Resampling.nearest)
                    nodata_ivm = src_ivm.nodata if src_ivm.nodata is not None else -9999.0
                else:
                    data_ivm = src_ivm.read(1)
                    nodata_ivm = src_ivm.nodata if src_ivm.nodata is not None else -9999.0
            valid_data_mask = (
                (data_ivm != nodata_ivm) & (~np.isnan(data_ivm)) & (data_ivm > -10000.0))
            valid_pixels = data_ivm[valid_data_mask]
            classifier = mapclassify.NaturalBreaks(valid_pixels, k=5)
            bins = classifier.bins
            threshold_safe = bins[0] if sample_mode == "ivm_very_low" else bins[1]
            self.log(f"  -> IVM阈值 <= {threshold_safe:.4f}")
            safe_area_mask = (data_ivm <= threshold_safe) & valid_data_mask

        self.log(f">>> [4/5] 生成正样本避让缓冲区 ({buffer_m}m)...")
        buffered_pos = gdf_pos.copy()
        buffered_pos['geometry'] = buffered_pos.geometry.buffer(buffer_m)
        valid_outside_pos = geometry_mask(
            geometries=buffered_pos.geometry, out_shape=shape,
            transform=transform, all_touched=True, invert=False)

        self.log(">>> [5/5] 抽取负样本...")
        candidate_pool = binary_erosion(
            safe_area_mask & valid_outside_pos & global_valid_mask, iterations=2)
        candidate_coords = np.argwhere(candidate_pool)
        self.log(f"  -> 候选像元: {len(candidate_coords)} 个")

        if len(candidate_coords) < target_n:
            self.log(f"⚠️ 候选像元不足，全部提取 ({len(candidate_coords)} 个)")
            target_n = len(candidate_coords)

        np.random.shuffle(candidate_coords)
        selected_pixels = []
        available_grid = candidate_pool.copy()
        pixel_size = abs(transform.a)
        radius_px = int(math.ceil(buffer_m / pixel_size))
        y, x = np.ogrid[-radius_px:radius_px + 1, -radius_px:radius_px + 1]
        circle_struct = x ** 2 + y ** 2 <= radius_px ** 2
        h, w = available_grid.shape

        for r, c in candidate_coords:
            if len(selected_pixels) >= target_n:
                break
            if available_grid[r, c]:
                selected_pixels.append((r, c))
                r_min, r_max = max(0, r - radius_px), min(h, r + radius_px + 1)
                c_min, c_max = max(0, c - radius_px), min(w, c + radius_px + 1)
                circ_r_min = radius_px - (r - r_min)
                circ_r_max = radius_px + (r_max - r)
                circ_c_min = radius_px - (c - c_min)
                circ_c_max = radius_px + (c_max - c)
                sub_circle = circle_struct[circ_r_min:circ_r_max, circ_c_min:circ_c_max]
                available_grid[r_min:r_max, c_min:c_max][sub_circle] = False

        self.log(f"  -> 成功提取: {len(selected_pixels)} 个负样本点位！")
        points = [
            Point(*rasterio.transform.xy(transform, r, c))
            for r, c in selected_pixels
        ]
        neg_gdf = gpd.GeoDataFrame(geometry=points, crs=crs)
        neg_gdf['Label'] = 0
        neg_gdf.to_file(out_shp_path, encoding='utf-8')
        self.log(f"✅ 负样本 SHP 写入成功: {out_shp_path}")


# ══════════════════════════════════════════════════════════════
# 工具函数
# ══════════════════════════════════════════════════════════════

def save_uploaded_tifs(uploaded_files, tmpdir):
    tif_dir = os.path.join(tmpdir, "factors")
    os.makedirs(tif_dir, exist_ok=True)
    for f in uploaded_files:
        with open(os.path.join(tif_dir, f.name), "wb") as out:
            out.write(f.read())
    return tif_dir


def save_uploaded_shp(zip_file, tmpdir, subfolder="shp"):
    shp_dir = os.path.join(tmpdir, subfolder)
    os.makedirs(shp_dir, exist_ok=True)
    with zipfile.ZipFile(zip_file) as zf:
        zf.extractall(shp_dir)
    shp_files = []
    for root, _, files in os.walk(shp_dir):
        shp_files += [os.path.join(root, f) for f in files if f.endswith('.shp')]
    if not shp_files:
        raise ValueError("ZIP 包中未找到 .shp 文件，请确认压缩包内容。")
    return shp_files[0]


def zip_output_dir(output_dir):
    zip_base = output_dir + "_results"
    shutil.make_archive(zip_base, 'zip', output_dir)
    with open(zip_base + ".zip", "rb") as f:
        return f.read()


# ══════════════════════════════════════════════════════════════
# Streamlit 界面
# ══════════════════════════════════════════════════════════════

def main():
    st.markdown("""
        <div style='background:#123a45;padding:24px 32px 18px;
                    border-radius:8px;margin-bottom:24px'>
            <h1 style='color:#f8fafc;margin:0;font-size:2rem'>
                🏔️ 滑坡易发性分析工具
            </h1>
            <p style='color:#cbd5e1;margin:6px 0 0'>
                Information Value Model · Spatial Sampling · GIS Workflow
            </p>
        </div>
    """, unsafe_allow_html=True)

    if not GIS_OK:
        st.error(
            f"❌ GIS 依赖库未安装，请先运行：\n\n"
            f"`pip install rasterio geopandas mapclassify shapely scipy`\n\n"
            f"错误: {GIS_ERR}")
        return

    tab_ivm, tab_sample = st.tabs(["📊 IVM 信息量评价", "📍 负样本生成"])

    # ── Tab 1: IVM 评价 ──────────────────────────────────────
    with tab_ivm:
        st.subheader("IVM 信息量评价")
        st.caption("上传环境因子 TIF 和滑坡正样本 SHP，自动计算 IV 值并生成综合易发性图。")

        col1, col2 = st.columns(2)
        with col1:
            tif_files = st.file_uploader(
                "📂 上传环境因子 TIF（可多选）",
                type=["tif", "tiff"],
                accept_multiple_files=True,
                key="ivm_tif"
            )
        with col2:
            shp_zip = st.file_uploader(
                "📌 上传滑坡正样本 SHP（打包为 ZIP）",
                type=["zip"],
                key="ivm_shp",
                help="将 .shp/.dbf/.shx/.prj 等文件一起打包成 ZIP 上传"
            )

        st.divider()

        if st.button("🚀 开始计算 IVM", type="primary", use_container_width=True):
            if not tif_files:
                st.error("请上传至少一个 TIF 因子文件。")
            elif shp_zip is None:
                st.error("请上传正样本 SHP（ZIP格式）。")
            else:
                log_box = st.empty()
                progress_bar = st.progress(0, text="准备中...")
                log_lines = []

                def log_fn(msg):
                    log_lines.append(msg)
                    log_box.text_area(
                        "运行日志", value="\n".join(log_lines),
                        height=260, key=f"ivm_log_{len(log_lines)}")

                def prog_fn(val):
                    progress_bar.progress(val, text=f"处理中... {int(val*100)}%")

                try:
                    with tempfile.TemporaryDirectory() as tmpdir:
                        tif_dir = save_uploaded_tifs(tif_files, tmpdir)
                        shp_path = save_uploaded_shp(shp_zip, tmpdir, "pos_shp")
                        output_dir = os.path.join(tmpdir, "ivm_output")
                        os.makedirs(output_dir, exist_ok=True)

                        service = GISAnalysisService(log_fn)
                        service.run_batch_analysis(
                            tif_dir, shp_path, output_dir, prog_fn)

                        progress_bar.progress(1.0, text="✅ 完成！")
                        st.success("IVM 计算完成！点击下方按钮下载所有结果。")

                        result_zip = zip_output_dir(output_dir)
                        st.download_button(
                            label="📥 下载全部结果（ZIP）",
                            data=result_zip,
                            file_name="IVM_Results.zip",
                            mime="application/zip",
                            use_container_width=True
                        )

                        csv_path = os.path.join(output_dir, "IV_Statistics.csv")
                        if os.path.exists(csv_path):
                            st.subheader("📋 IV 统计表预览")
                            st.dataframe(
                                pd.read_csv(csv_path), use_container_width=True)

                except Exception as e:
                    st.error(f"运行出错：{e}")
                    st.code(traceback.format_exc())

    # ── Tab 2: 负样本生成 ────────────────────────────────────
    with tab_sample:
        st.subheader("负样本生成")
        st.caption(
            "基于 IVM 易发性分区和正样本缓冲区，自动生成空间分布合理的负样本点。")

        col1, col2 = st.columns(2)
        with col1:
            s_tif_files = st.file_uploader(
                "📂 上传环境因子 TIF（可多选）",
                type=["tif", "tiff"],
                accept_multiple_files=True,
                key="sample_tif"
            )
            s_pos_shp = st.file_uploader(
                "📌 上传滑坡正样本 SHP（ZIP）",
                type=["zip"],
                key="sample_pos_shp"
            )
        with col2:
            s_ivm_tif = st.file_uploader(
                "🗺️ 上传 IVM 易发性结果图 TIF（IVM策略必填）",
                type=["tif", "tiff"],
                key="sample_ivm"
            )
            sample_mode = st.selectbox(
                "采样策略",
                options=["ivm_low", "ivm_very_low", "random"],
                format_func=lambda x: {
                    "ivm_low":      "极低 + 低易发区（推荐）",
                    "ivm_very_low": "仅极低易发区",
                    "random":       "全区随机"
                }[x]
            )
            buffer_m = st.number_input(
                "避让缓冲距离（米）", value=500, min_value=0, step=50)

        st.divider()

        if st.button("🚀 开始生成负样本", type="primary", use_container_width=True):
            if not s_tif_files:
                st.error("请上传环境因子 TIF 文件。")
            elif s_pos_shp is None:
                st.error("请上传正样本 SHP（ZIP格式）。")
            elif sample_mode in ["ivm_low", "ivm_very_low"] and s_ivm_tif is None:
                st.error("IVM 约束策略需要上传 IVM 易发性结果图 TIF。")
            else:
                log_box2 = st.empty()
                log_lines2 = []

                def log_fn2(msg):
                    log_lines2.append(msg)
                    log_box2.text_area(
                        "运行日志", value="\n".join(log_lines2),
                        height=260, key=f"sample_log_{len(log_lines2)}")

                try:
                    with tempfile.TemporaryDirectory() as tmpdir:
                        tif_dir = save_uploaded_tifs(s_tif_files, tmpdir)
                        shp_path = save_uploaded_shp(s_pos_shp, tmpdir, "pos_shp2")
                        out_shp_path = os.path.join(tmpdir, "negative_samples.shp")

                        ivm_tif_path = ""
                        if s_ivm_tif is not None:
                            ivm_tif_path = os.path.join(tmpdir, "ivm_result.tif")
                            with open(ivm_tif_path, "wb") as f:
                                f.write(s_ivm_tif.read())

                        service = NegativeSampleService(log_fn2)
                        service.generate_negative_samples(
                            factor_dir=tif_dir,
                            ivm_tif_path=ivm_tif_path,
                            pos_shp_path=shp_path,
                            out_shp_path=out_shp_path,
                            sample_mode=sample_mode,
                            buffer_m=int(buffer_m)
                        )

                        st.success("负样本生成完成！")

                        shp_zip_path = os.path.join(tmpdir, "negative_samples_shp.zip")
                        with zipfile.ZipFile(shp_zip_path, 'w') as zf:
                            for ext in ['.shp', '.dbf', '.shx', '.prj', '.cpg']:
                                fp = out_shp_path.replace('.shp', ext)
                                if os.path.exists(fp):
                                    zf.write(fp, os.path.basename(fp))
                        with open(shp_zip_path, "rb") as f:
                            shp_zip_buf = f.read()

                        st.download_button(
                            label="📥 下载负样本 SHP（ZIP）",
                            data=shp_zip_buf,
                            file_name="negative_samples.zip",
                            mime="application/zip",
                            use_container_width=True
                        )

                except Exception as e:
                    st.error(f"运行出错：{e}")
                    st.code(traceback.format_exc())

    st.divider()
    st.caption(
        "滑坡易发性分析工具 v4.5 Web版 · 基于 IVM 信息量模型 · Powered by Streamlit")


if __name__ == "__main__":
    main()