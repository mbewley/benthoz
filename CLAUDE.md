# BENTHOZ-2015 toolbox

Tools to access, manipulate and classify the BENTHOZ-2015 data set: ~10,000 AUV
images of the Australian sea floor, with expert annotations on over 400,000
image points, labelled against the CATAMI hierarchy of ~148 classes.

Cite the [Scientific Data paper](http://www.nature.com/articles/sdata201557) if
you use the data set.

## Layout

| Path | What |
| --- | --- |
| `prep/label_converter.py` | Builds the CATAMI class hierarchy from `idlookups.csv` |
| `prep/labels.py` | Sparse point labels; rasterises masks on demand |
| `prep/patches.py` | Extracts square patches around annotated points |
| `prep/image_fetcher.py` | Downloads the source GeoTIFFs from S3 |
| `prepare_dataset.ipynb` | End-to-end preparation pipeline |
| `data_splits/` | Published train/cal/test point labels (the source of record) |
| `idlookups.csv` | Class id → CATAMI name, 148 classes |
| `BENTHOZ-2015-imagelist.csv` | Image manifest, 9,882 rows, drives the downloader |
| `testing_data/` | Small checked-in fixtures for the test suite |

## Data

The imagery is **not** in the repo. It is expected at `benthoz_data/`, which is
gitignored — symlink it to wherever the data actually lives:

```bash
ln -s /path/to/benthoz_data benthoz_data
```

```
benthoz_data/
  images/            9,874 GeoTIFFs, 1360x1024, already JPEG-compressed internally
  labels/            train.parquet, cal.parquet, test.parquet
```

Note the images are JPEG-compressed *inside* the TIFF container. Re-exporting
them to `.jpg` produces **larger** files and adds generation loss — don't.

OpenCV emits `TIFFReadDirectory: Unknown field with tag 34735/34737/34264`
warnings on read. Those are GeoTIFF geo-referencing tags it doesn't parse. Harmless.

## Labels

Annotations are **sparse points** — roughly 40 per image — not dense masks.

Storage is one parquet file per split, ~1.4 MB for all three. Writing needs no
custom code:

```python
prep.labels.load_points("data_splits/public_labels_train.csv").to_parquet(path, index=False)
```

`load_points` clips the ~23 published points that sit one pixel outside the image
(row 1024, col 1360).

A dense `(n_classes, height, width)` mask is built **on demand**, never stored:

```python
mask = prep.labels.points_to_mask(points_for_one_image, df_id_lookups)
```

An earlier version of the pipeline stored those masks as `.npz` per image. Avoid
going back to that: each array is 206 MB in memory to carry ~50 meaningful values,
and it cost 263 MB on disk for the training split alone.

`points_to_mask` reproduces those old arrays exactly. The `.npz` files are gone, so
that equivalence is now guarded by `reference_mask` in `tests/prep/test_labels.py` —
a verbatim copy of the original per-point loop, kept as an independent oracle. **Do
not "tidy it up" to call `points_to_mask`**; the whole point is that the two
implementations are written differently and must agree.

For patch classifiers, skip masks entirely and use
`prep.labels.points_to_bit_vectors(points, df_id_lookups)` — one target vector
per annotated point.

### Hierarchy semantics

Each class has a bit vector over all classes, with three states:

- `YES` (1) — the class itself and all its ancestors
- `DONT_KNOW` (-1) — its descendants; a point labelled at an internal node says
  nothing about which child applies
- `NO` (0) — everything else

Unannotated pixels in a mask are `DONT_KNOW` throughout.

Two gotchas in `build_hierarchy_from_id_lookup`:

- `Biota` and `Physical` are not prefixed onto their children's names in
  `idlookups.csv`, so they are prepended manually via a hardcoded list of their
  children. A new top-level child would need adding to that list.
- Ancestor lookup resolves parents by name prefix and will `IndexError` if a
  parent row is absent. Any cut-down lookup table must be **closed under
  ancestors** — see `testing_data/idlookups_test.csv`.

## Development

Python 3.14+.

```bash
pip install -e ".[dev,notebook]"
pytest
```

`pygraphviz` is deliberately **not** in the `notebook` extra — it compiles against
system headers and breaks the install on a machine without them. It lives in its
own extra:

```bash
sudo apt install graphviz graphviz-dev && pip install -e ".[graphviz]"
```

Only the notebook's hierarchy plot needs it; nothing else does.

The suite runs from any working directory — `pyproject.toml` anchors the pytest
rootdir and the root `conftest.py` provides path-anchored fixtures. Don't
reintroduce `../`-relative data paths in tests.

Formatting is black with `line-length = 120`.

Docker is the reliable way to get the full stack, including GPU and graphviz:

```bash
docker build -t benthoz .
```

### There is no CI

Nothing runs the tests for you. **Run `pytest` locally before pushing or opening a
PR** — that is the only gate this repo has. If a change touches `prep/labels.py`,
`prep/label_converter.py`, or the class hierarchy, say so in the PR, because a
regression there is silent in the data rather than loud in a crash.

### Testing notes

`tests/prep/test_label_converter.py` and `test_labels.py` run against the 14-class
fixture in `testing_data/idlookups_test.csv`, which spans both roots and depths
1–6. Regenerate it by taking a set of leaves plus all their ancestors from the
full `idlookups.csv`.

The channel-mean assertions in `test_patches.py` are regression guards on pixel
content; if they move, patch extraction changed.

Watch for assertions that cannot fail. The original suite had four written
`assert x.shape, (127, 127, 3)` — a message argument, not a comparison — which
passed unconditionally for years. When adding a guard, break the code on purpose
once and confirm the test goes red.
