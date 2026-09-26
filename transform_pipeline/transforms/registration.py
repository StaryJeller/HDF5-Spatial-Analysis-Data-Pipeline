from collections.abc import Iterable
import numpy as np
from transform_pipeline.core.models import ShiftMeasurement,RegistrationResult
def corr(a,b):
    a=np.asarray(a,float)-np.mean(a);b=np.asarray(b,float)-np.mean(b);d=np.sqrt(np.sum(a*a)*np.sum(b*b));return np.nan if d<=1e-12 else float(np.sum(a*b)/d)
def score(moving,reference,shift,margin):
    x=np.arange(moving.size,dtype=float);s=np.interp(x-shift,x,moving,left=np.nan,right=np.nan);a=s[margin:-margin];b=reference[margin:-margin];v=np.isfinite(a)&np.isfinite(b);return np.nan if v.sum()<3 else corr(a[v],b[v])
def smooth(values,weights,sigma):
    r=max(1,int(np.ceil(4*sigma)));o=np.arange(-r,r+1);k=np.exp(-.5*(o/sigma)**2);n=np.convolve(values*weights,k,'full');d=np.convolve(weights,k,'full');start=(k.size-1)//2;n=n[start:start+values.size];d=d[start:start+values.size];out=np.full(values.shape,np.nan);v=d>1e-12;out[v]=n[v]/d[v];return out
def measure_subpixel_shifts(total_counts,minimum_shift=-2,maximum_shift=2,shift_step=.025,edge_margin=16,smoothing_sigma_odd_rows=5,minimum_weight=.0005):
    rows,cols=total_counts.shape
    if 2*edge_margin>=cols: raise ValueError("Edge margin removes row")
    candidates=np.arange(minimum_shift,maximum_shift+.5*shift_step,shift_step);zi=int(np.argmin(np.abs(candidates)));odd=np.arange(1,rows-1,2);raw=[];peaks=[];zeros=[];gains=[]
    for r in odd:
        m=total_counts[r].astype(float);ref=.5*(total_counts[r-1].astype(float)+total_counts[r+1].astype(float));scores=np.array([score(m,ref,float(s),edge_margin) for s in candidates]);i=int(np.nanargmax(scores));shift=float(candidates[i]);peak=float(scores[i])
        if 0<i<len(scores)-1:
            l,c,rr=scores[i-1:i+2];den=l-2*c+rr
            if np.isfinite(den) and abs(den)>1e-12:
                off=float(np.clip(.5*(l-rr)/den,-1,1));shift+=off*shift_step;peak=float(c-.25*(l-rr)*off)
        zero=float(scores[zi]);raw.append(shift);peaks.append(peak);zeros.append(zero);gains.append(peak-zero)
    raw=np.asarray(raw);gains=np.asarray(gains);sm=smooth(raw,np.maximum(gains,minimum_weight),smoothing_sigma_odd_rows)
    return [ShiftMeasurement(int(r),float(a),float(b),float(p),float(z),float(g)) for r,a,b,p,z,g in zip(odd,raw,sm,peaks,zeros,gains)]
def shift_spectral_row(row,shift,max_abs=1):
    if not np.isfinite(shift) or abs(shift)>max_abs: raise ValueError(f"Invalid shift {shift}")
    cols,ch=row.shape;dest=np.arange(cols,dtype=float);pos=dest-shift;valid=(pos>=0)&(pos<=cols-1);out=np.full((cols,ch),np.nan,np.float32);p=pos[valid];left=np.floor(p).astype(int);right=np.minimum(left+1,cols-1);wr=(p-left).astype(np.float32);src=np.asarray(row,np.float32);out[valid]=src[left]*(1-wr)[:,None]+src[right]*wr[:,None];return out,valid
def apply_subpixel_registration(cube,shifts:Iterable[ShiftMeasurement],maximum_absolute_shift=1):
    rows,cols,ch=cube.shape;lst=list(shifts);by={s.row:s for s in lst};expected=set(range(1,rows-1,2))
    if set(by)!=expected: raise ValueError("Shift coverage mismatch")
    out=np.full((rows,cols,ch),np.nan,np.float32);mask=np.zeros((rows,cols),bool)
    for r in range(rows):
        if r%2==0: out[r]=cube[r].astype(np.float32);mask[r]=True
        else: out[r],mask[r]=shift_spectral_row(cube[r],by[r].smoothed_shift,maximum_absolute_shift)
    total=np.nansum(out,axis=2,dtype=np.float64).astype(np.float32);total[~mask]=np.nan;return RegistrationResult(out,total,mask,lst)
