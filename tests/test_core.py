import numpy as np
from transform_pipeline.transforms.raster import deserpentine,infer_serpentine_geometry
def test_deserpentine(): assert np.array_equal(deserpentine(np.array([[0,1,2],[5,4,3]])),np.array([[0,1,2],[3,4,5]]))
def test_geometry():
    x=np.array([0,1,2,2,1,0]);y=np.array([0,0,0,1,1,1]);g=infer_serpentine_geometry(x,y);assert (g.rows,g.columns)==(2,3)
