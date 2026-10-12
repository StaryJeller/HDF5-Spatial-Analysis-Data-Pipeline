# Four-quadrant playtest contract

Place two or more quadrant acquisition directories under this directory. Each
quadrant directory must contain:

```text
one *.h5 file
sdd1_0.bin
sdd2_0.bin
sdd3_0.bin
sdd4_0.bin
mcc_flyer_0.csv        # optional acquisition sidecar
```

Example local layout:

```text
Testmodel_4_quadrant/
├── 150721_NCo_7_SW/
├── 152250_NCo_7_SE/
├── 153821_NCo_7_NE/
└── 155353_NCo_7_NW/
```

Per-quadrant `prototype_output/` and root `stitched_output/` content are local and
ignored by Git.
