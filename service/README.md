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

One case at a time (`service/inference_gate.py`, shared with the Sybil and CVD services); a case
running longer than `INFERENCE_MAX_SECONDS` (30 min) makes the process exit so Docker restarts it.

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
slices at the same position. No slice analysable → 422.

## Version

`cad@src.<code>+w.<weights>+clf.<fingerprint>` — code of `modules/inference.py`, `service/app.py`,
`service/case.py` and the feature settings; the U-Net file and the two training CSVs; the
classifier's predictions on its whole training set (it is re-trained in memory at every start,
with `random_state=42`, so a change of library version shows up as a different fingerprint).

## Build and run

```sh
git lfs pull                                   # the U-Net weights (373 MB)
docker build -f service/Dockerfile -t cad-service .
docker run --rm -p 5557:5557 -v <uploads>:/app/uploads -v <results>:/app/results cad-service
```

CPU only in this release (`tensorflow-cpu`): TensorFlow would otherwise take most of the GPU
memory the Sybil and CVD services share. In the diagnosis app the service is started through
the `cad` compose profile — see its `SETUPDOCKER.md`.

## Tests

`python -m pytest service/tests` — the case layer (`service/case.py`), no TensorFlow needed.
