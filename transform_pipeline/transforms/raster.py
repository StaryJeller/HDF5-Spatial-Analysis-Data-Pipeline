import numpy as np
from transform_pipeline.core.models import RasterGeometry
def deserpentine(a):
    out=np.array(a,copy=True);out[1::2]=out[1::2,::-1,...];return out
def infer_serpentine_geometry(x,y,tolerance=1e-9):
    if x.ndim!=1 or y.ndim!=1 or x.shape!=y.shape: raise ValueError(f"Invalid coordinate shapes: X={x.shape}, Y={y.shape}")
    b=np.concatenate(([0],np.flatnonzero(np.abs(np.diff(y))>tolerance)+1,[len(y)]));lengths=np.diff(b)
    if lengths.size==0 or not np.all(lengths==lengths[0]): raise ValueError("Raster rows are not uniform")
    rows,cols=int(lengths.size),int(lengths[0]);xr=x.reshape(rows,cols);yr=y.reshape(rows,cols)
    if not np.allclose(yr,yr[:,:1],rtol=0,atol=tolerance): raise ValueError("Y varies within a row")
    if not np.all(np.diff(yr[:,0])>0): raise ValueError("Y rows are not increasing")
    for i,row in enumerate(xr):
        dx=np.diff(row);ok=np.all(dx>0) if i%2==0 else np.all(dx<0)
        if not ok: raise ValueError(f"Unexpected X direction in row {i}")
    xg,yg=deserpentine(xr),deserpentine(yr);xa,ya=xg[0].copy(),yg[:,0].copy()
    if not np.allclose(xg,xa[None,:],rtol=0,atol=tolerance): raise ValueError("Irregular X grid")
    if not np.allclose(yg,ya[:,None],rtol=0,atol=tolerance): raise ValueError("Irregular Y grid")
    return RasterGeometry(rows,cols,xa,ya)
