# Data

The dataset is **not** included. Download PAD-UFES-20 from Mendeley Data:

- https://data.mendeley.com/datasets/zr7vgbcyr2/1  (DOI 10.17632/zr7vgbcyr2.1)

Extract it so the folder looks like this:

```
data/pad-ufes-20/
├── metadata.csv
└── images/
    ├── imgs_part_1/PAT_*.png
    ├── imgs_part_2/PAT_*.png
    └── imgs_part_3/PAT_*.png
```

Any sub-folder layout under `images/` works: images are found recursively and
matched to `metadata.csv` through the `img_id` column. You can also point the
pipeline elsewhere with `python run_pipeline.py --data-dir /path/to/pad-ufes-20`.
