import os
import numpy as np
import pandas as pd
import anndata
from tifffile import TiffFile, imread, imwrite  # ✅ <-- must be here
from skimage.measure import regionprops_table, label
from skimage.filters import gaussian, threshold_otsu
from skimage.morphology import remove_small_objects, binary_opening, disk
from bigfish.detection import detect_spots
import matplotlib.pyplot as plt
from tifffile import imread
from skimage.measure import find_contours
from skimage.exposure import rescale_intensity
from cellpose import models


def load_nd_metadata(path):
    metadata = {}
    with open(path, "r") as file:
        for line in file:
            parts = line.strip().split(",", 1)
            if len(parts) == 2:
                key = parts[0].strip('" ')
                value = parts[1].strip('" ')
                metadata[key] = value
    return metadata

def extract_tiff_metadata(filepath):
    with TiffFile(filepath) as tif:
        tags = tif.pages[0].tags
        shape = tif.series[0].shape
        desc = tags.get('ImageDescription')
        uic1 = tags.get('UIC1tag')

        meta = {
            "image_shape_z": shape[0] if len(shape) == 3 else None,
            "image_shape_y": shape[1] if len(shape) >= 2 else None,
            "image_shape_x": shape[2] if len(shape) == 3 else None,
            "bits_per_sample": tags.get("BitsPerSample", None).value if tags.get("BitsPerSample") else None,
            "exposure": None,
            "XCalibration": None,
            "YCalibration": None,
            "CalibrationUnits": None,
        }

        if desc:
            val = desc.value
            if isinstance(val, bytes):
                val = val.decode('utf-8', errors='ignore')
            for line in val.split('\n'):
                if "Exposure" in line:
                    meta["exposure"] = line.strip()
                    break

        if uic1 and isinstance(uic1.value, dict):
            meta.update({
                "XCalibration": uic1.value.get("XCalibration"),
                "YCalibration": uic1.value.get("YCalibration"),
                "CalibrationUnits": uic1.value.get("CalibrationUnits")
            })
    return meta

def generate_mips(stack1_path, stack2_path, output_folder, sample_name):
    stack1 = imread(stack1_path)
    stack2 = imread(stack2_path)
    mip1 = stack1.max(axis=0)
    mip2 = stack2.max(axis=0)

    mip1_path = os.path.join(output_folder, f"{sample_name}_mip_channel1.tif")
    mip2_path = os.path.join(output_folder, f"{sample_name}_mip_channel2.tif")

    imwrite(mip1_path, mip1.astype(np.uint16))
    imwrite(mip2_path, mip2.astype(np.uint16))
    
    return mip1, mip2, mip1_path, mip2_path

def run_cellpose_segmentation(mip, model_type='cyto3'):
    model = models.Cellpose(gpu=False, model_type='cyto3')
    mip_smoothed = gaussian(mip, sigma=1)
    masks, _, _, _ = model.eval(mip_smoothed, diameter=None, channels=[0, 0])
    return masks

def extract_nuclei_properties(masks, mip, sample_name, x_cal, y_cal):
    labeled = label(masks)
    props = regionprops_table(
        labeled,
        intensity_image=mip,
        properties=['label', 'area', 'mean_intensity', 'max_intensity', 'min_intensity',
                    'centroid', 'bbox']
    )
    props_df = pd.DataFrame(props)

    X_cols = ['area', 'mean_intensity', 'max_intensity', 'min_intensity']
    X = props_df[X_cols].values.astype(np.float32)

    obs = props_df.drop(columns=X_cols)
    obs["Object_Tag"] = "nuclei"
    obs["sample_name"] = sample_name

    if x_cal and y_cal:
        obs["area_µm2"] = X[:, X_cols.index("area")] * (x_cal * y_cal)
    else:
        obs["area_µm2"] = np.nan

    return X, obs, X_cols

def build_anndata(X, obs, var_names, metadata, paths, mask_path):
    adata = anndata.AnnData(X=X, obs=obs, var=pd.DataFrame(index=var_names))
    adata.uns["sample_metadata"] = metadata
    adata.uns.update(paths)
    adata.uns["nuclei_mask_path"] = mask_path
    return adata

def segment_cytoplasm_by_cellpose(mip, diameter=250):
    model = models.Cellpose(gpu=False, model_type='cyto3')
    masks, _, _, _ = model.eval(mip, diameter=diameter, channels=[0, 0])
    return masks

def detect_smfish_spots(mip, voxel_size_nm=(100, 100), spot_radius_nm=(150, 150),
                        spot_radius_px=3, cell_mask=None, save_dot_mask_path=None):
    import bigfish.stack as stack
    import bigfish.detection as detection
    from skimage.draw import disk
    from skimage.measure import regionprops, label, regionprops_table
    import pandas as pd
    import numpy as np

    # Estimate detection radius in pixels
    radius_est = detection.get_object_radius_pixel(voxel_size_nm, spot_radius_nm, ndim=2)

    # Filter and detect
    rna_log = stack.log_filter(mip, sigma=radius_est)
    maxima = detection.local_maximum_detection(rna_log, min_distance=radius_est)
    threshold = detection.automated_threshold_setting(rna_log, maxima)
    spots, _ = detection.spots_thresholding(rna_log, maxima, threshold)

    if spots.shape[0] == 0:
        return pd.DataFrame()

    # Build dot mask with 2px radius
    dot_mask = np.zeros(mip.shape, dtype=np.uint16)
    for idx, (y, x) in enumerate(spots, start=1):
        rr, cc = disk((y, x), radius=spot_radius_px, shape=mip.shape)
        dot_mask[rr, cc] = idx

    if save_dot_mask_path:
        from tifffile import imwrite
        imwrite(save_dot_mask_path, dot_mask)

    # Extract spot features
    regions = regionprops(label(dot_mask), intensity_image=mip)
    records = []
    for region in regions:
        records.append({
            "label": region.label,
            "area": region.area,
            "min_intensity": region.min_intensity,
            "max_intensity": region.max_intensity,
            "mean_intensity": region.mean_intensity,
            "total_intensity": region.intensity_image.sum(),
            "centroid-0": region.centroid[0],
            "centroid-1": region.centroid[1],
            "Object_Tag": "smFISH_spot"
        })

    df = pd.DataFrame(records)

    # ----- Exclude spot regions when computing background -----
    if cell_mask is not None:
        labeled_cells = label(cell_mask)

        # Build binary dot mask (spots only)
        dot_mask_binary = np.zeros(mip.shape, dtype=bool)
        for y, x in spots:
            rr, cc = disk((y, x), radius=spot_radius_px, shape=mip.shape)
            dot_mask_binary[rr, cc] = True

        # Remove spots from the cytoplasm mask
        cell_mask_cleaned = labeled_cells.copy()
        cell_mask_cleaned[dot_mask_binary] = 0  # set spot pixels to background

        # Compute background stats only on non-spot cytoplasm regions
        props = regionprops_table(
            cell_mask_cleaned,
            intensity_image=mip,
            properties=["area", "mean_intensity"]
        )
        df_cells = pd.DataFrame(props)
        df_cells = df_cells[df_cells["area"] > 100]

        background_mean = df_cells["mean_intensity"].mean()
        df["background_mean_intensity"] = background_mean
        df["signal_to_background"] = df["mean_intensity"] / background_mean

    return df

def visualize_anndata_segmentation_with_spots(h5ad_path):
    adata = anndata.read_h5ad(h5ad_path)
    uns = adata.uns

    # Load MIPs and masks
    mip1 = imread(uns.get("mip_path_channel1")) if "mip_path_channel1" in uns else None
    mip2 = imread(uns.get("mip_path_channel2")) if "mip_path_channel2" in uns else None
    nuclei_mask = imread(uns.get("nuclei_mask_path")) if "nuclei_mask_path" in uns else None
    cell_mask = imread(uns.get("cytoplasm_mask_path")) if "cytoplasm_mask_path" in uns else None
    spot_mask = imread(uns.get("smfish_spot_mask_path")) if "smfish_spot_mask_path" in uns else None

    def enhance(image):
        low, high = np.percentile(image, (1, 99))
        return rescale_intensity(image, in_range=(low, high), out_range=(0, 255)).astype(np.uint8)

    mip1 = enhance(mip1) if mip1 is not None else None
    mip2 = enhance(mip2) if mip2 is not None else None

    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    # Subplot 1: smFISH + Cell contours
    if mip1 is not None and cell_mask is not None:
        axs[0].imshow(mip1, cmap="gray")
        for contour in find_contours(cell_mask, 0.5):
            axs[0].plot(contour[:, 1], contour[:, 0], linewidth=1)
        axs[0].set_title("smFISH + Cell contours")
        axs[0].axis("off")

    # Subplot 2: Nuclei + Nuclei contours
    if mip2 is not None and nuclei_mask is not None:
        axs[1].imshow(mip2, cmap="gray")
        for contour in find_contours(nuclei_mask, 0.5):
            axs[1].plot(contour[:, 1], contour[:, 0], linewidth=1)
        axs[1].set_title("Nuclei + Nuclei contours")
        axs[1].axis("off")

    # Subplot 3: smFISH + Spot locations
    if mip1 is not None and spot_mask is not None:
        axs[2].imshow(mip1, cmap="gray")
        for contour in find_contours(spot_mask, 0.5):
            axs[2].plot(contour[:, 1], contour[:, 0], linewidth=1)
        axs[2].set_title("smFISH + Spot contours")
        axs[2].axis("off")

    plt.tight_layout()
    plt.show()

def plot_spot_signal_to_background(df_spots, sample_name=None, snr_thresholds=(1, 2, 3), show=True, save_path=None):
    """
    Plot histogram of smFISH spot signal-to-background ratios (SNR).

    Parameters:
        df_spots (pd.DataFrame): DataFrame with 'signal_to_background' column.
        sample_name (str): Optional name for plot title.
        snr_thresholds (tuple): Thresholds to mark on the plot.
        show (bool): Whether to display the plot interactively.
        save_path (str or None): If given, save the plot to this file.
    """
    if "signal_to_background" not in df_spots.columns:
        raise ValueError("Missing 'signal_to_background' column in input DataFrame.")

    plt.figure(figsize=(8, 5))
    snr_values = df_spots["signal_to_background"]
    plt.hist(snr_values, bins=100, edgecolor='black', alpha=0.8)

    plt.xlabel("Signal-to-Background Ratio (SNR)")
    plt.ylabel("Number of Spots")
    title = "smFISH Spot SNR Distribution"
    if sample_name:
        title += f" - {sample_name}"
    plt.title(title)

    # Draw threshold lines
    for val in snr_thresholds:
        plt.axvline(val, color='red', linestyle='--', label=f"SNR = {val}")

    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"📊 Saved SNR plot to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()

def plot_spot_snr_heatmap(mip, spots_df, sample_name=None, save_path=None, colormap='plasma', vmax=None):
    """
    Plot smFISH MIP with overlaid spots colored by SNR (signal-to-background ratio).

    Parameters:
        mip (ndarray): 2D smFISH MIP image.
        spots_df (pd.DataFrame): DataFrame with spot centroids and signal_to_background.
        sample_name (str): Optional, title of the plot.
        save_path (str): Optional path to save the image.
        colormap (str): Matplotlib colormap name.
        vmax (float or None): Optional max value for color scaling.
    """
    import matplotlib.pyplot as plt

    if "signal_to_background" not in spots_df.columns:
        raise ValueError("Missing 'signal_to_background' column in DataFrame.")

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(mip, cmap='gray')

    # Normalize colormap range
    snr_values = spots_df["signal_to_background"].clip(lower=0)
    vmax = vmax or snr_values.quantile(0.98)  # clip high outliers for better contrast

    scatter = ax.scatter(
        spots_df["centroid-1"],  # x (columns)
        spots_df["centroid-0"],  # y (rows)
        c=snr_values,
        cmap=colormap,
        s=30,
        edgecolor='black',
        linewidth=0.5,
        vmin=0,
        vmax=vmax
    )

    plt.colorbar(scatter, ax=ax, label="Signal-to-Background Ratio (SNR)")
    title = "smFISH Spot SNR Heatmap"
    if sample_name:
        title += f" - {sample_name}"
    ax.set_title(title)
    ax.axis("off")

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📸 Saved SNR heatmap to: {save_path}")

    plt.show()



from matplotlib.backends.backend_pdf import PdfPages
from skimage.exposure import rescale_intensity
from skimage.measure import find_contours
import matplotlib.pyplot as plt

def generate_segmentation_report(
    sample_name, mip1, mip2,
    nuclei_mask, cytoplasm_mask, spot_mask,
    spots_before_filtering, spots_after_filtering,
    snr_values, output_path, crop_coords=(800, 1200, 800, 1200)
):
    with PdfPages(output_path) as pdf:
        # Enhance image for display
        def enhance(img):
            low, high = np.percentile(img, (1, 99))
            return rescale_intensity(img, in_range=(low, high), out_range=(0, 255)).astype(np.uint8)

        mip1_disp = enhance(mip1)
        mip2_disp = enhance(mip2)

        # --- Overview of segmentation ---
        fig, axs = plt.subplots(1, 3, figsize=(18, 6))

        axs[0].imshow(mip1_disp, cmap='gray')
        if cytoplasm_mask is not None:
            for contour in find_contours(cytoplasm_mask, 0.5):
                axs[0].plot(contour[:, 1], contour[:, 0], linewidth=1)
        axs[0].set_title("smFISH + Cytoplasm Mask")
        axs[0].axis("off")

        axs[1].imshow(mip2_disp, cmap='gray')
        if nuclei_mask is not None:
            for contour in find_contours(nuclei_mask, 0.5):
                axs[1].plot(contour[:, 1], contour[:, 0], linewidth=1)
        axs[1].set_title("Nuclei + Nuclei Mask")
        axs[1].axis("off")

        axs[2].imshow(mip1_disp, cmap='gray')
        if spot_mask is not None:
            for contour in find_contours(spot_mask, 0.5):
                axs[2].plot(contour[:, 1], contour[:, 0], linewidth=1)
        axs[2].set_title("smFISH + Spot Mask")
        axs[2].axis("off")

        plt.tight_layout()
        pdf.savefig()
        plt.close()

        # --- Histogram of SNR values ---
        fig = plt.figure(figsize=(8, 5))
        plt.hist(snr_values, bins=100, edgecolor='black', alpha=0.8)
        for t in [1, 2, 3]:
            plt.axvline(t, linestyle='--', color='red', label=f"SNR = {t}")
        plt.title("Spot SNR Distribution")
        plt.xlabel("Signal-to-Background Ratio")
        plt.ylabel("Spot Count")
        plt.legend()
        plt.tight_layout()
        pdf.savefig()
        plt.close()

        # --- Cropped spot detection before and after SNR filtering ---
        y0, y1, x0, x1 = crop_coords
        crop = mip1_disp[y0:y1, x0:x1]

        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        axs[0].imshow(crop, cmap='gray')
        axs[0].set_title("Cropped Spot Detection (Before SNR Filter)")
        axs[1].imshow(crop, cmap='gray')
        axs[1].set_title("Cropped Spot Detection (After SNR ≥ 3)")

        # Hollow circles before filtering
        for coord in spots_before_filtering:
            y, x = coord[0] - y0, coord[1] - x0
            if 0 <= y < crop.shape[0] and 0 <= x < crop.shape[1]:
                circ = plt.Circle((x, y), radius=3, fill=False, edgecolor='red', linewidth=1)
                axs[0].add_patch(circ)

        # Hollow circles after filtering
        for _, row in spots_after_filtering.iterrows():
            y, x = row["centroid-0"] - y0, row["centroid-1"] - x0
            if 0 <= y < crop.shape[0] and 0 <= x < crop.shape[1]:
                circ = plt.Circle((x, y), radius=3, fill=False, edgecolor='lime', linewidth=1)
                axs[1].add_patch(circ)

        for ax in axs:
            ax.axis("off")

        plt.tight_layout()
        pdf.savefig()
        plt.close()

    print(f"📄 Report saved: {output_path}")

import os
import numpy as np
import pandas as pd
import anndata
from tifffile import imread
from skimage.measure import label, regionprops, find_contours
from matplotlib.patches import Circle
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

def analyze_sample(adata_path: str, snr_threshold: float = 3.0):
    """
    Analyze a single .h5ad file and generate summary tables and a PDF report.

    Parameters:
        adata_path (str): Full path to the .h5ad file for one sample.
        snr_threshold (float): Minimum signal-to-background ratio to consider a spot.
    """
    # Load data
    if not os.path.exists(adata_path):
        raise FileNotFoundError(f"H5AD file not found: {adata_path}")
    
    adata = anndata.read_h5ad(adata_path)
    sample_name = adata.uns["sample_metadata"]["sample_name"]
    output_dir = os.path.join(os.path.dirname(adata_path))
    os.makedirs(output_dir, exist_ok=True)

    # Load masks and MIPs
    cell_mask = imread(adata.uns["cytoplasm_mask_path"])
    nuclei_mask = imread(adata.uns["nuclei_mask_path"])
    mip1 = imread(adata.uns["mip_path_channel1"])
    mip2 = imread(adata.uns["mip_path_channel2"])
    height, width = cell_mask.shape

    # Label masks
    labeled_cells = label(cell_mask)
    labeled_nuclei = label(nuclei_mask)
    num_cells_total = labeled_cells.max()
    num_nuclei_total = labeled_nuclei.max()

    # Filter valid nuclei (not at image border)
    valid_nucleus_labels = [
        prop.label for prop in regionprops(labeled_nuclei)
        if prop.bbox[0] > 0 and prop.bbox[1] > 0 and prop.bbox[2] < height and prop.bbox[3] < width
    ]

    # Match valid nuclei to cells
    valid_cells = set()
    for nucleus_label in valid_nucleus_labels:
        nucleus_mask = labeled_nuclei == nucleus_label
        overlapping = np.unique(labeled_cells[nucleus_mask])
        overlapping = overlapping[overlapping > 0]
        valid_cells.update(overlapping)
    valid_cells = sorted(valid_cells)

    # Extract high-SNR smFISH spots
    spots_df = adata.obs[
        (adata.obs["Object_Tag"] == "smFISH_spot") &
        (adata.obs["signal_to_background"] > snr_threshold)
    ].copy()

    # Spot locations
    ys = spots_df["centroid-0"].round().astype(int).clip(0, height - 1)
    xs = spots_df["centroid-1"].round().astype(int).clip(0, width - 1)
    spots_df["cell_label"] = labeled_cells[ys, xs]
    spots_df["nucleus_label"] = labeled_nuclei[ys, xs]

    # Filter to valid cells
    spots_df = spots_df[spots_df["cell_label"].isin(valid_cells)]
    spots_df["location"] = np.where(spots_df["nucleus_label"] > 0, "nucleus", "cytoplasm")

    # Spot count per cell
    if spots_df.empty:
        counts = pd.DataFrame(columns=["cell_label", "spots_in_nucleus", "spots_in_cytoplasm"]).astype({
            "cell_label": int, "spots_in_nucleus": int, "spots_in_cytoplasm": int
        })
    else:
        counts = (
            spots_df.groupby(["cell_label", "location"]).size().unstack(fill_value=0)
            .rename(columns={"nucleus": "spots_in_nucleus", "cytoplasm": "spots_in_cytoplasm"})
            .reset_index()
        )

    all_valid_cells = pd.DataFrame({"cell_label": valid_cells})
    counts = all_valid_cells.merge(counts, on="cell_label", how="left")

    # Ensure columns exist even if there were zero spots
    for col in ["spots_in_nucleus", "spots_in_cytoplasm"]:
        if col not in counts.columns:
            counts[col] = 0

    counts = counts.fillna({"spots_in_nucleus": 0, "spots_in_cytoplasm": 0})
    counts["spots_in_nucleus"] = counts["spots_in_nucleus"].astype(int)
    counts["spots_in_cytoplasm"] = counts["spots_in_cytoplasm"].astype(int)

    # Save table
    counts_path = os.path.join(output_dir, f"{sample_name}_cellwise_spot_counts.csv")
    counts.to_csv(counts_path, index=False)

    # Summary statistics
    stats = {
        "Sample name": sample_name,
        "Total nuclei": num_nuclei_total,
        "Total cells": num_cells_total,
        "Valid cells (with full nucleus)": len(valid_cells),
        "Total smFISH spots": len(adata.obs[adata.obs["Object_Tag"] == "smFISH_spot"]),
        f"High SNR spots (> {snr_threshold})": len(spots_df),
        "Spots in nucleus": int((spots_df["location"] == "nucleus").sum()),
        "Spots in cytoplasm": int((spots_df["location"] == "cytoplasm").sum())
    }

    stats_df = pd.DataFrame(list(stats.items()), columns=["Metric", "Value"])
    stats_path = os.path.join(output_dir, f"{sample_name}_summary_statistics.csv")
    stats_df.to_csv(stats_path, index=False)

    # PDF Visualization
    def normalize(img):
        p1, p99 = np.percentile(img, (1, 99.9))
        return np.clip((img - p1) / (p99 - p1), 0, 1)

    mip1_norm = normalize(mip1)
    mip2_norm = normalize(mip2)
    merged_rgb = np.stack([mip1_norm, mip1_norm, mip2_norm], axis=-1)

    pdf_path = os.path.join(output_dir, f"{sample_name}_real_cells_with_spots.pdf")
    with PdfPages(pdf_path) as pdf:
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(merged_rgb)

        for cell_label in valid_cells:
            cell_binary = (labeled_cells == cell_label).astype(np.uint8)
            for contour in find_contours(cell_binary, 0.5):
                ax.plot(contour[:, 1], contour[:, 0], linewidth=1, color='white')

        for _, row in spots_df.iterrows():
            circ = Circle((row["centroid-1"], row["centroid-0"]), radius=3,
                          edgecolor='red', facecolor='none', linewidth=1.5, alpha=0.8)
            ax.add_patch(circ)

        ax.set_title(f"{sample_name} – Valid Cells (white) with High-SNR smFISH Spots (red)", fontsize=14)
        ax.axis("off")
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    print(f"✅ Summary CSV: {stats_path}")
    print(f"✅ Cellwise Spot Table: {counts_path}")
    print(f"✅ PDF Report: {pdf_path}")


def process_sample(folder_path: str, sample_name: str, crop_coords=(800, 1200, 800, 1200), snr_threshold: float = 3.0):
    """
    Run segmentation and extract features for a single sample.

    Parameters:
        folder_path (str): Path to folder containing .nd and .stk files.
        sample_name (str): Base name for the sample (e.g. "sample1").
    """

    # ---- Setup ----
    output_dir = os.path.join(folder_path, "analysis_output")
    os.makedirs(output_dir, exist_ok=True)
    print(f"📁 Processing: {sample_name}")

    nd_path = os.path.join(folder_path, f"{sample_name}.nd")
    stk1_path = os.path.join(folder_path, f"{sample_name}_w1conf561Virtex.stk")
    stk2_path = os.path.join(folder_path, f"{sample_name}_w2conf405Virtex.stk")

    # ---- Metadata ----
    nd_meta = load_nd_metadata(nd_path)
    tiff_meta = extract_tiff_metadata(stk1_path)

    # ---- MIPs ----
    mip1, mip2, mip1_path, mip2_path = generate_mips(stk1_path, stk2_path, output_dir, sample_name)

    # ---- Segment ----
    cytoplasm_mask = segment_cytoplasm_by_cellpose(mip1)
    cyto_mask_path = os.path.join(output_dir, f"{sample_name}_cytoplasm_mask.tif")
    imwrite(cyto_mask_path, cytoplasm_mask.astype(np.uint16))

    nuclei_mask = run_cellpose_segmentation(mip2, model_type='cyto3')
    nuclei_mask_path = os.path.join(output_dir, f"{sample_name}_nuclei_mask.tif")
    imwrite(nuclei_mask_path, nuclei_mask.astype(np.uint16))

    # ---- Extract Nuclei Features ----
    X_nuclei, obs_nuclei, X_cols = extract_nuclei_properties(
        nuclei_mask, mip2, sample_name,
        x_cal=tiff_meta["XCalibration"], y_cal=tiff_meta["YCalibration"]
    )

    # ---- Detect smFISH Spots ----
    dot_mask_path = os.path.join(output_dir, f"{sample_name}_dot_mask.tif")
    spots_df = detect_smfish_spots(
        mip1,
        voxel_size_nm=(tiff_meta["XCalibration"] * 1000, tiff_meta["YCalibration"] * 1000),
        spot_radius_nm=(150, 150),
        spot_radius_px=2,
        cell_mask=cytoplasm_mask,
        save_dot_mask_path=dot_mask_path
    )

    # ---- Organize Metadata ----
    sample_meta = {
        "sample_name": sample_name,
        "z_slices": int(nd_meta.get("NZSteps", 0)),
        "z_step_size": float(nd_meta.get("ZStepSize", 0.0)),
        "num_wavelengths": int(nd_meta.get("NWavelengths", 0)),
        "channel_1_name": nd_meta.get("WaveName1"),
        "channel_2_name": nd_meta.get("WaveName2"),
        "start_time": nd_meta.get("StartTime1"),
        "image_shape_z": tiff_meta["image_shape_z"],
        "image_shape_y": tiff_meta["image_shape_y"],
        "image_shape_x": tiff_meta["image_shape_x"],
        "bits_per_sample": tiff_meta["bits_per_sample"],
        "exposure": tiff_meta["exposure"],
        "x_calibration": tiff_meta["XCalibration"],
        "y_calibration": tiff_meta["YCalibration"],
        "calibration_units": tiff_meta["CalibrationUnits"]
    }

    path_meta = {
        "nd_path": nd_path,
        "stk_path_channel1": stk1_path,
        "stk_path_channel2": stk2_path,
        "mip_path_channel1": mip1_path,
        "mip_path_channel2": mip2_path,
        "cytoplasm_mask_path": cyto_mask_path,
        "nuclei_mask_path": nuclei_mask_path,
        "smfish_spot_mask_path": dot_mask_path
    }

    # ---- Build AnnData ----
    if not spots_df.empty:
        X_spots = spots_df[["area", "mean_intensity", "max_intensity", "min_intensity", "total_intensity"]].values
        obs_spots = spots_df[["label", "Object_Tag", "centroid-0", "centroid-1"]].copy()
        obs_spots["sample_name"] = sample_name
        obs_spots["area_µm2"] = np.nan
        if "signal_to_background" in spots_df.columns:
            obs_spots["signal_to_background"] = spots_df["signal_to_background"].values

        adata_spots = anndata.AnnData(
            X=X_spots,
            obs=obs_spots,
            var=pd.DataFrame(index=["area", "mean_intensity", "max_intensity", "min_intensity", "total_intensity"])
        )

        adata_nuclei = anndata.AnnData(X=X_nuclei, obs=obs_nuclei, var=pd.DataFrame(index=pd.Index(X_cols, dtype=str)))
        adata = adata_nuclei.concatenate(adata_spots, index_unique=None)
        if "batch" in adata.obs.columns:
            adata.obs.drop(columns=["batch"], inplace=True)
    else:
        adata = anndata.AnnData(X=X_nuclei, obs=obs_nuclei, var=pd.DataFrame(index=X_cols))

    adata.uns["sample_metadata"] = sample_meta
    adata.uns.update(path_meta)

    # ---- Save AnnData ----
    output_h5ad = os.path.join(output_dir, f"{sample_name}_nuclei_spots.h5ad")
    adata.write(output_h5ad)
    print(f"💾 Saved AnnData to: {output_h5ad}")


    # ---- PDF Report ----
    report_path = os.path.join(output_dir, f"{sample_name}_segmentation_report.pdf")

    crop_coords = crop_coords
    y0, y1, x0, x1 = crop_coords

    # ---- Crop raw spots ----
    spots_raw_coords = spots_df[["centroid-0", "centroid-1"]].values.astype(int) if not spots_df.empty else np.empty((0, 2))

    # ---- Filtered spots ----
    snr_filtered = spots_df[
        (spots_df["signal_to_background"] >= snr_threshold) &
        (spots_df["centroid-0"].between(y0, y1)) &
        (spots_df["centroid-1"].between(x0, x1))
    ] if not spots_df.empty else pd.DataFrame(columns=spots_df.columns)

    # ---- Get SNR values ----
    snr_vals = spots_df["signal_to_background"].dropna().values if "signal_to_background" in spots_df.columns else np.array([])

    generate_segmentation_report(
        sample_name=sample_name,
        mip1=mip1, mip2=mip2,
        nuclei_mask=nuclei_mask,
        cytoplasm_mask=cytoplasm_mask,
        spot_mask=imread(dot_mask_path) if os.path.exists(dot_mask_path) else None,
        spots_before_filtering=spots_raw_coords,
        spots_after_filtering=snr_filtered,
        snr_values=snr_vals,
        output_path=report_path,
        crop_coords=crop_coords
    )

    print(f"✅ Finished sample: {sample_name}\n")


import xml.etree.ElementTree as ET

def load_ome_metadata(path):
    metadata = {}
    try:
        tree = ET.parse(path)
        root = tree.getroot()

        # Extract relevant metadata from OME root (namespaces are common in OME-XML)
        ns = {'ome': 'http://www.openmicroscopy.org/Schemas/OME/2016-06'}

        image_elem = root.find('ome:Image', ns)
        if image_elem is not None:
            pixels = image_elem.find('ome:Pixels', ns)
            if pixels is not None:
                metadata["SizeZ"] = pixels.attrib.get("SizeZ")
                metadata["SizeY"] = pixels.attrib.get("SizeY")
                metadata["SizeX"] = pixels.attrib.get("SizeX")
                metadata["SizeT"] = pixels.attrib.get("SizeT")
                metadata["PhysicalSizeX"] = pixels.attrib.get("PhysicalSizeX")
                metadata["PhysicalSizeY"] = pixels.attrib.get("PhysicalSizeY")
                metadata["PhysicalSizeZ"] = pixels.attrib.get("PhysicalSizeZ")
                metadata["PhysicalSizeXUnit"] = pixels.attrib.get("PhysicalSizeXUnit")
                metadata["PhysicalSizeYUnit"] = pixels.attrib.get("PhysicalSizeYUnit")
                metadata["PhysicalSizeZUnit"] = pixels.attrib.get("PhysicalSizeZUnit")

    except ET.ParseError as e:
        print(f"❌ Error parsing OME XML: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

    return metadata

def process_sample_Merian_OME(
    folder_path: str,
    sample_name: str,
    ome_file: str,
    channel1_file: str,
    channel2_file: str,
    crop_coords=(800, 1200, 800, 1200),
    snr_threshold: float = 3.0
):
    """
    Run segmentation and extract features for a single sample.

    Parameters:
        folder_path (str): Path to folder containing sample files.
        sample_name (str): Name of the sample.
        ome_file (str): File name of the .companion.ome metadata file.
        channel1_file (str): File name of the channel 1 .ome.tif image.
        channel2_file (str): File name of the channel 2 .ome.tif image.
    """

    # ---- Setup ----
    output_dir = os.path.join(folder_path, "analysis_output")
    os.makedirs(output_dir, exist_ok=True)
    print(f"📁 Processing: {sample_name}")

    nd_path = os.path.join(folder_path, ome_file)
    stk1_path = os.path.join(folder_path, channel1_file)
    stk2_path = os.path.join(folder_path, channel2_file)

    # ---- Metadata ----
    ome_meta = load_ome_metadata(nd_path)
    tiff_meta = extract_tiff_metadata(stk1_path)

    # ---- MIPs ----
    mip1, mip2, mip1_path, mip2_path = generate_mips(stk1_path, stk2_path, output_dir, sample_name)

    # ---- Segment ----
    cytoplasm_mask = segment_cytoplasm_by_cellpose(mip1)
    cyto_mask_path = os.path.join(output_dir, f"{sample_name}_cytoplasm_mask.tif")
    imwrite(cyto_mask_path, cytoplasm_mask.astype(np.uint16))

    nuclei_mask = run_cellpose_segmentation(mip2, model_type='cyto3')
    nuclei_mask_path = os.path.join(output_dir, f"{sample_name}_nuclei_mask.tif")
    imwrite(nuclei_mask_path, nuclei_mask.astype(np.uint16))

    # ---- Extract Nuclei Features ----
    X_nuclei, obs_nuclei, X_cols = extract_nuclei_properties(
        nuclei_mask, mip2, sample_name,
        x_cal=float(ome_meta["PhysicalSizeX"]), y_cal=float(ome_meta["PhysicalSizeY"])
    )

    # ---- Detect smFISH Spots ----
    dot_mask_path = os.path.join(output_dir, f"{sample_name}_dot_mask.tif")
    spots_df = detect_smfish_spots(
        mip1,
        voxel_size_nm=(float(ome_meta["PhysicalSizeX"]) * 1000, float(ome_meta["PhysicalSizeY"]) * 1000),
        spot_radius_nm=(150, 150),
        spot_radius_px=2,
        cell_mask=cytoplasm_mask,
        save_dot_mask_path=dot_mask_path
    )

    # ---- Organize Metadata ----
    sample_meta = {
        "sample_name": sample_name,
        "z_slices": int(ome_meta.get("NZSteps", 0)),
        "z_step_size": float(ome_meta.get("ZStepSize", 0.0)),
        "num_wavelengths": int(ome_meta.get("NWavelengths", 0)),
        "channel_1_name": ome_meta.get("WaveName1"),
        "channel_2_name": ome_meta.get("WaveName2"),
        "start_time": ome_meta.get("StartTime1"),
        "image_shape_z": tiff_meta["image_shape_z"],
        "image_shape_y": tiff_meta["image_shape_y"],
        "image_shape_x": tiff_meta["image_shape_x"],
        "bits_per_sample": tiff_meta["bits_per_sample"],
        "exposure": tiff_meta["exposure"],
        "x_calibration": float(ome_meta["PhysicalSizeX"]),
        "y_calibration": float(ome_meta["PhysicalSizeY"]),
        "calibration_units": tiff_meta["CalibrationUnits"]
    }

    path_meta = {
        "nd_path": nd_path,
        "stk_path_channel1": stk1_path,
        "stk_path_channel2": stk2_path,
        "mip_path_channel1": mip1_path,
        "mip_path_channel2": mip2_path,
        "cytoplasm_mask_path": cyto_mask_path,
        "nuclei_mask_path": nuclei_mask_path,
        "smfish_spot_mask_path": dot_mask_path
    }

    # ---- Build AnnData ----
    if not spots_df.empty:
        X_spots = spots_df[["area", "mean_intensity", "max_intensity", "min_intensity", "total_intensity"]].values
        obs_spots = spots_df[["label", "Object_Tag", "centroid-0", "centroid-1"]].copy()
        obs_spots["sample_name"] = sample_name
        obs_spots["area_µm2"] = np.nan
        if "signal_to_background" in spots_df.columns:
            obs_spots["signal_to_background"] = spots_df["signal_to_background"].values

        adata_spots = anndata.AnnData(
            X=X_spots,
            obs=obs_spots,
            var=pd.DataFrame(index=["area", "mean_intensity", "max_intensity", "min_intensity", "total_intensity"])
        )

        adata_nuclei = anndata.AnnData(X=X_nuclei, obs=obs_nuclei, var=pd.DataFrame(index=pd.Index(X_cols, dtype=str)))
        adata = adata_nuclei.concatenate(adata_spots, index_unique=None)
        if "batch" in adata.obs.columns:
            adata.obs.drop(columns=["batch"], inplace=True)
    else:
        adata = anndata.AnnData(X=X_nuclei, obs=obs_nuclei, var=pd.DataFrame(index=X_cols))

    adata.uns["sample_metadata"] = sample_meta
    adata.uns.update(path_meta)

    # ---- Save AnnData ----
    output_h5ad = os.path.join(output_dir, f"{sample_name}_nuclei_spots.h5ad")
    adata.write(output_h5ad)
    print(f"💾 Saved AnnData to: {output_h5ad}")

    # ---- PDF Report ----
    report_path = os.path.join(output_dir, f"{sample_name}_segmentation_report.pdf")

    y0, y1, x0, x1 = crop_coords

    spots_raw_coords = spots_df[["centroid-0", "centroid-1"]].values.astype(int) if not spots_df.empty else np.empty((0, 2))

    snr_filtered = spots_df[
        (spots_df["signal_to_background"] >= snr_threshold) &
        (spots_df["centroid-0"].between(y0, y1)) &
        (spots_df["centroid-1"].between(x0, x1))
    ] if not spots_df.empty else pd.DataFrame(columns=spots_df.columns)

    snr_vals = spots_df["signal_to_background"].dropna().values if "signal_to_background" in spots_df.columns else np.array([])

    generate_segmentation_report(
        sample_name=sample_name,
        mip1=mip1, mip2=mip2,
        nuclei_mask=nuclei_mask,
        cytoplasm_mask=cytoplasm_mask,
        spot_mask=imread(dot_mask_path) if os.path.exists(dot_mask_path) else None,
        spots_before_filtering=spots_raw_coords,
        spots_after_filtering=snr_filtered,
        snr_values=snr_vals,
        output_path=report_path,
        crop_coords=crop_coords
    )

    print(f"✅ Finished sample: {sample_name}\n")

