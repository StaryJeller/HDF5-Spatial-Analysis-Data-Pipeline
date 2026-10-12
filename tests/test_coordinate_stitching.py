from pathlib import Path
import numpy as np
import xarray as xr

from transform_pipeline.stitching import build_stitch_plan, stitch_canonical_stores


def make_tile(path: Path, x, y, value: float, include_spectral=True):
    x=np.asarray(x,dtype=float); y=np.asarray(y,dtype=float); channel=np.arange(3,dtype=np.int32)
    total=np.full((y.size,x.size),value,dtype=np.float32)
    variables={
        "combined_total_counts_registered":(("y","x"),total),
        "combined_validity_mask":(("y","x"),np.ones_like(total,dtype=bool)),
    }
    if include_spectral:
        variables["combined_counts_registered"]=(("y","x","channel"),np.repeat(total[:,:,None],3,axis=2))
    ds=xr.Dataset(variables,coords={"y":y,"x":x,"channel":channel})
    ds.to_zarr(path,mode="w",consolidated=True,zarr_format=2)


def test_coordinate_plan_and_mean_overlap(tmp_path: Path):
    first=tmp_path/"tile_a"/"canonical_sgm.zarr"; first.parent.mkdir()
    second=tmp_path/"tile_b"/"canonical_sgm.zarr"; second.parent.mkdir()
    make_tile(first,[0,1,2],[0,1],2.0)
    make_tile(second,[2,3,4],[0,1],4.0)
    plan=build_stitch_plan([first,second])
    assert plan.x.tolist()==[0,1,2,3,4]
    assert [(p.x_start,p.x_stop) for p in plan.placements]==[(0,3),(2,5)]

    output=tmp_path/"mosaic.zarr"
    stitch_canonical_stores([first,second],output,"mean",True,overwrite=True)
    ds=xr.open_zarr(output,consolidated=True,chunks=None)
    try:
        np.testing.assert_allclose(ds.mosaic_total_counts_registered.values,[[2,2,3,4,4],[2,2,3,4,4]])
        assert ds.overlap_count.values[0].tolist()==[1,1,2,1,1]
        assert ds.source_tile_index.values[0].tolist()==[0,0,0,1,1]
        assert ds.mosaic_counts_registered.shape==(2,5,3)
    finally: ds.close()


def test_first_policy_keeps_first_overlap(tmp_path: Path):
    first=tmp_path/"a.zarr"; second=tmp_path/"b.zarr"
    make_tile(first,[0,1],[0,1],5.0,False)
    make_tile(second,[1,2],[0,1],9.0,False)
    output=tmp_path/"first.zarr"
    stitch_canonical_stores([first,second],output,"first",False,overwrite=True)
    ds=xr.open_zarr(output,consolidated=True,chunks=None)
    try:
        np.testing.assert_allclose(ds.mosaic_total_counts_registered.values,[[5,5,9],[5,5,9]])
    finally: ds.close()
