import argparse
from pathlib import Path
from transform_pipeline.io.readers.cls_sgm import default_playtest_scan_dir
from transform_pipeline.pipeline import process_sdd_scan,write_pipeline_outputs
def main():
    p=argparse.ArgumentParser();p.add_argument('scan_dir',type=Path,nargs='?',default=None,help='Defaults to Playtest_model/Testmodel_NCo_7_SW');p.add_argument('--h5');p.add_argument('--sdd',default='sdd1_0.bin');p.add_argument('--output-dir',type=Path);p.add_argument('--save-cubes',action='store_true');a=p.parse_args();scan=a.scan_dir or default_playtest_scan_dir();out=a.output_dir or scan/'prototype_output';rec,reg=process_sdd_scan(scan,a.sdd,a.h5);result=write_pipeline_outputs(rec,reg,out,a.save_cubes);print(f'Pipeline complete. Output: {result.output_dir}')
if __name__=='__main__':main()
