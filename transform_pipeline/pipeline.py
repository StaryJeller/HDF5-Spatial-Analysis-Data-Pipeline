from dataclasses import dataclass
from pathlib import Path
import numpy as np
from transform_pipeline.io.readers.cls_sgm import resolve_scan_dir,find_hdf5_file,reconstruct_scan
from transform_pipeline.transforms.registration import measure_subpixel_shifts,apply_subpixel_registration
from transform_pipeline.io.writers.tiff import *
@dataclass(frozen=True)
class PipelineOutputs:
    output_dir:Path;float_tiff:Path;preview_png:Path;validity_mask:Path;shift_csv:Path
def process_sdd_scan(scan_dir=None,sdd_filename='sdd1_0.bin',h5_filename=None):
    scan_dir=resolve_scan_dir(scan_dir);h5=find_hdf5_file(scan_dir,h5_filename);binary=scan_dir/sdd_filename
    if not binary.is_file(): raise FileNotFoundError(binary)
    rec=reconstruct_scan(h5,binary);shifts=measure_subpixel_shifts(rec.total_counts);return rec,apply_subpixel_registration(rec.cube,shifts)
def write_pipeline_outputs(rec,reg,output_dir,save_cubes=False):
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    if save_cubes: np.save(out/'sdd_cube_reconstructed.npy',rec.cube);np.save(out/'sdd_cube_subpixel_registered.npy',reg.cube)
    np.save(out/'total_counts_subpixel_registered.npy',reg.total_counts);np.save(out/'x_axis.npy',rec.geometry.x_axis);np.save(out/'y_axis.npy',rec.geometry.y_axis)
    tif=out/'total_counts_subpixel_registered_float32.tif';png=out/'total_counts_subpixel_registered_preview.png';mask=out/'subpixel_validity_mask.tif';csvp=out/'subpixel_shifts.csv'
    write_float_tiff(tif,reg.total_counts,'Subpixel-registered SDD total counts');write_validity_mask(mask,reg.validity_mask);save_preview(png,make_uint8_preview(reg.total_counts,reg.validity_mask));write_shift_csv(csvp,reg.shifts);return PipelineOutputs(out,tif,png,mask,csvp)
