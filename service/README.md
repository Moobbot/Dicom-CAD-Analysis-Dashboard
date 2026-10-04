# CAD-COVID model service

An HTTP wrapper around `modules/inference.py` for the hospital diagnosis app
(`dicom-diagnosis`). It scores one CT case and answers with the app's **result contract v1**.

> **Research model — not validated for clinical use.** The served pipeline (U-Net lung mask →
> 24 radiomics features → random forest) has never been evaluated on CT series: the classifier
> is trained on features extracted from 2D JPG images, and no reference case with a known answer
> exists. The diagnosis app keeps this model **disabled** until the operator declares its intended
> use and enables it. Its numbers must not be used to diagnose or treat patients.

## API

| Route | Answer |
|---|---|
| `GET /health` | `{status, model_loaded, busy, waiting, running_seconds, stuck}`; 503 with a `reason` while the model is not usable. Answers while a case runs. |
| `GET /info` | `{model: "cad", loaded, version, weights, result_contract: 1, research_model: true, ...}` |
| `POST /api_predict` | body `{session_id, output_dir}`; reads `UPLOAD_FOLDER/<session_id>/`, writes `RESULTS_FOLDER/<session_id>/<output_dir>/` |

One case at a time in this service (`service/inference_gate.py`, the same code as the Sybil and CVD
services — each service has its own lock; the app's "one model at a time" mode is what keeps the
three from running together). A case running longer than `INFERENCE_MAX_SECONDS` (30 min) makes the
process exit so Docker restarts it.

## What a case returns

The model scores 2D slices. The case follows the original rule (`predict.py`): **COVID-19 if any
slice is classified COVID-19.** The service adds three safeguards:

- **No silent fallback.** The original code quietly switches to a threshold mask or to made-up
  features when the U-Net or PyRadiomics fail, and still reports "success". Here the service
  refuses to start without them (`/health` says why), and a slice where a fallback happened at
  run time is **not analysed**: the result says how many slices were left out, and why.
- **No conclusion from partial coverage.** Slices left out and no positive slice among the others
  → **"Indeterminate"**, never "Normal".
- **Nothing identifying leaves the service.** Slices are read in z order from their DICOM
  headers (never by file name), the GIF frames are labelled "slice i/N", no DICOM tag is returned.

Results (`kind`): `predicted_class` (label), `mean_covid_score` (score — the mean random-forest
vote share, **not a calibrated probability**), `covid_slices` and `analysed_slices`
(measurements, in slices). Artifact: `results.gif` (lung mask overlay, ≤ 60 frames).

Refused cases (400): no DICOM image, a non-CT modality, a multi-frame file, several series, two
slices with the same position and the same instance number. Skipped: non-DICOM files and DICOM
objects that are not images (DICOMDIR, reports). A `.dcm` file that cannot be read counts as a slice
not analysed. No slice analysable → 422, with the reasons.

Measured on the project's 291-slice reference CT: the U-Net returns an **empty lung mask on 76
slices, many of them in the middle of the lungs** (not only above and below them), so those slices
are not analysed and a case without a positive slice is "Indeterminate". This is the model's
behaviour on hospital CT, not a wrapper choice — one more reason it is a research model.

## Version

`cad@src.<code>+w.<weights>+clf.<fingerprint>` — code of `modules/inference.py`, `service/app.py`,
`service/case.py` and the feature settings; the U-Net file and the two training CSVs; the
classifier's predictions on its whole training set (it is re-trained in memory at every start,
with `random_state=42`, so a change of library version shows up as a different fingerprint).

The `w.` part covers the name of each file as well as its content. The U-Net downloaded by the
service (`lung_segmentation_unet.h5`) therefore gives another `w.` than the same file from Git LFS
(`lung_segmentation_unet .h5`): measured `w.2d0ba4955fd8` and `w.09a63383b647`. `/info` lists the
SHA-256 of each file, which is the same in both cases.

## Build and run

```sh
git lfs pull                                   # the U-Net weights (373 MB)
docker build -f service/Dockerfile -t cad-service .
docker run --rm -p 5557:5557 -v "$PWD/ML/UNET Training:/app/ML/UNET Training" \
  -v <uploads>:/app/uploads -v <results>:/app/results cad-service
```

The image does not contain the weights: the service reads them from the folder mounted at
`/app/ML/UNET Training`.

CPU only in this release (`tensorflow-cpu`): TensorFlow would otherwise take most of the GPU
memory the Sybil and CVD services share. In the diagnosis app the service is built and started
with the other services (no compose profile); the model stays disabled and hidden until an
administrator enables it — see the app's `SETUPDOCKER.md`.

## Weights

`service/weights.py`, at every start:

- A real HDF5 file under `lung_segmentation_unet .h5` (the name in git, with a space) or
  `lung_segmentation_unet.h5` is used as it is. The model takes the first of the two names that
  holds a real HDF5 file, so a Git LFS pointer under the first name does not hide a real file
  under the second. When that file is not the one the service was validated with (another U-Net,
  or a copy that was cut short), the log says `present but NOT the validated file`; if it cannot
  be loaded, `/health` names it and says to move it away.
- Otherwise (no file, or only a Git LFS pointer because the checkout was made without git-lfs)
  the U-Net is downloaded from this repository's release `v-0.1` to `lung_segmentation_unet.h5`
  and installed **only if its SHA-256 is the expected one**. The address needs `github.com` and
  `release-assets.githubusercontent.com`.
- **Nothing in the folder is ever replaced or removed**: not the pointer (the checkout stays
  clean; the downloaded name is in `.gitignore`), not a file under the download's name, not a
  file that appears while the download runs. The one exception is the service's own temporary
  file of a download that was killed (`lung_segmentation_unet.h5.<8 characters>.partial`),
  removed at the next start.
- The download runs before the service starts answering: on a slow connection the container
  shows `unhealthy` until it has finished (the log says `downloading`). Wait; a restart starts
  the download again.
- When the file cannot be provided, `/health` answers 503 with the reason and the log says what
  to do (`git lfs pull`, or copy the file into the folder, then restart).

The log has one line per start, for example
`Weights: lung_segmentation_unet.h5 present; predictions.npy present.`

## Tests

`python -m pytest service/tests` — the case layer (`service/case.py`), the weights
(`service/weights.py`) and the choice of the U-Net file: no TensorFlow needed. The parity test
(`test_parity.py`) needs the image's libraries and a case: see its docstring.
