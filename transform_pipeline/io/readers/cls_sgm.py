from pathlib import Path
import h5py, numpy as np
from transform_pipeline.core.models import ReconstructionResult
from transform_pipeline.transforms.raster import infer_serpentine_geometry,deserpentine
def repository_root(): return Path(__file__).resolve().parents[3]
def default_playtest_scan_dir(): return repository_root()/"Playtest_model"/"Testmodel_NCo_7_SW"
def resolve_scan_dir(scan_dir=None):
    path=default_playtest_scan_dir() if scan_dir is None else Path(scan_dir)
    path=path.resolve()
    if not path.is_dir(): raise FileNotFoundError(f"Scan directory not found: {path}")
    return path
def find_hdf5_file(scan_dir,filename=None):
    scan_dir=resolve_scan_dir(scan_dir)
    if filename:
        p=scan_dir/filename
        if not p.is_file(): raise FileNotFoundError(p)
        return p
    found=sorted(scan_dir.glob('*.h5'))
    if len(found)!=1: raise ValueError(f"Expected exactly one .h5 in {scan_dir}; found {len(found)}")
    return found[0]
def load_coordinates(path):
    with h5py.File(path,'r') as h5:
        return np.asarray(h5['/hexapod_waves/x'][:],dtype=float),np.asarray(h5['/hexapod_waves/y'][:],dtype=float)
def read_sdd_cube(path,geometry,channels=256,dtype='<u4'):
    raw=np.fromfile(path,dtype=np.dtype(dtype));expected=geometry.rows*geometry.columns*channels
    if raw.size!=expected: raise ValueError(f"Expected {expected} values, found {raw.size}")
    return deserpentine(raw.reshape(geometry.rows,geometry.columns,channels))
def reconstruct_scan(h5_path,bin_path,channels=256,dtype='<u4'):
    x,y=load_coordinates(h5_path);g=infer_serpentine_geometry(x,y);cube=read_sdd_cube(bin_path,g,channels,dtype)
    return ReconstructionResult(cube,cube.sum(axis=2,dtype=np.uint64),g)
