from transform_pipeline.io.readers.cls_sgm import repository_root,default_playtest_scan_dir
def test_playtest_path_is_repository_relative():
    assert default_playtest_scan_dir()==repository_root()/"Playtest_model"/"Testmodel_NCo_7_SW"
