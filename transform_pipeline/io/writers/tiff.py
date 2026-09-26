from pathlib import Path
import csv,numpy as np,tifffile
from PIL import Image
def write_float_tiff(path,image,description):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tifffile.imwrite(path,np.asarray(image,np.float32),photometric='minisblack',metadata={'axes':'YX','description':description})
def write_validity_mask(path,mask): tifffile.imwrite(path,mask.astype(np.uint8)*255,photometric='minisblack',metadata={'axes':'YX'})
def make_uint8_preview(image,mask=None,low_percentile=1,high_percentile=99):
    mask=np.isfinite(image) if mask is None else mask;v=image[mask&np.isfinite(image)];lo,hi=np.percentile(v,[low_percentile,high_percentile]);scaled=np.clip((image.astype(float)-lo)/(hi-lo),0,1);scaled=np.nan_to_num(scaled);out=np.rint(scaled*255).astype(np.uint8);out[~mask|~np.isfinite(image)]=0;return out
def save_preview(path,preview): Image.fromarray(preview,mode='L').save(path)
def write_shift_csv(path,shifts):
    with Path(path).open('w',encoding='utf-8',newline='') as f:
        fields=['row','raw_fractional_shift','smoothed_fractional_shift','peak_correlation','zero_shift_correlation','correlation_gain'];w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for s in shifts:w.writerow(dict(zip(fields,[s.row,s.raw_shift,s.smoothed_shift,s.peak_correlation,s.zero_correlation,s.correlation_gain])))
