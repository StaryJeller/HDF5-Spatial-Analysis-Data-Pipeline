from transform_pipeline.io.readers.cls_sgm import default_playtest_scan_dir
from transform_pipeline.pipeline import process_sdd_scan,write_pipeline_outputs
scan=default_playtest_scan_dir();reconstruction,registration=process_sdd_scan();outputs=write_pipeline_outputs(reconstruction,registration,scan/'prototype_output',save_cubes=True);print(outputs)
