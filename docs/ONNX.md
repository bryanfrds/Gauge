# The ONNX backend

## What this is

PyTorch is a framework for *training* models. YN never trains at runtime — it only runs
a model that is already trained — so PyTorch's training machinery is loaded and paid for
without ever being used.

ONNX (Open Neural Network Exchange) is a portable file format for a trained model. ONNX
Runtime is a small program that runs one. Exporting YN's model to ONNX lets it skip
PyTorch entirely at runtime.

## Using it

```
pip install 'yn[onnx,export]'   # export needs torch too; running does not
yn export-onnx
```

The export lands in `~/.cache/yn/onnx/<model>/` and is picked up automatically.

| Variable | Effect |
|---|---|
| `YN_BACKEND` | `auto` (default: ONNX when an export exists, unless `YN_DEVICE` names a GPU), `torch`, or `onnx` |
| `YN_ONNX_DIR` | Where exports live. Point this at a mounted volume in a container. |
| `YN_ONNX_THREADS` | Cap ONNX Runtime's threads (a whole number, 1 or more), so several YN processes on one small box don't each grab every core. A bad value is an error, even on `auto`. |

`YN_BACKEND=onnx` fails loudly when there is no export. `auto` quietly falls back to
PyTorch, which is what you want on a developer machine.

`yn export-onnx --out <dir>` only writes into an empty directory or over an earlier
export. It refuses a directory holding anything else, and it never deletes a path it
didn't create. YN looks for an export in a folder named after the model, so if you
export elsewhere, name the folder as the command tells you and set `YN_ONNX_DIR` to
its parent.

## Measurements

Default model (`MoritzLaurer/deberta-v3-base-zeroshot-v2.0`), `YN_DEVICE=cpu`, one
process per row. The workload is four short inputs through both `check_many` and
`decide_many` — sixteen text/statement pairs in total.

| | Peak RSS | Load | Inference |
|---|---|---|---|
| PyTorch | 709 MB | 4.0 s | 0.99 s |
| ONNX Runtime | 608 MB | 0.6 s | 0.27 s |

Measured separately, the import cost alone is 329 MB for `torch` + `transformers`
against 46 MB for `onnxruntime` + `tokenizers`. Most of the saving is not loading
PyTorch at all; the rest of the resident total is the weights, which both backends pay.

## Agreement with PyTorch

Measured over short and long inputs, through both `check` and `decide`, the largest
difference in `confidence` was **0.0013**; treat 0.002 as the bound.

The two backends are not bit-identical — the graph is optimized differently and
operations are fused differently — so this is float drift, not a bug. It is far below
what the model's own calibration means, and the chosen answer never changed in testing.
The one place it can show is `sure`: a confidence within 0.002 of `YN_THRESHOLD` may
fall either side of it. If an exact match matters more than the memory, use
`YN_BACKEND=torch`.

That bound isn't just prose — `tests/test_backend_agreement.py` enforces it against
the real model. It is skipped by default because it needs both backends and an export
on disk:

```
YN_SLOW_TESTS=1 python -m pytest tests/test_backend_agreement.py
```

If it fails, the backends have diverged or the number here is wrong. Don't just raise
the constant.

## If ONNX can't load

`auto` falls back to PyTorch and says nothing — set `YN_VERBOSE=1` to see why. This
matters because `pip install 'yn[export]'` alone gives you enough to *produce* an
export but not to *run* one, and a complete export with no runtime would otherwise
fail every call. `YN_BACKEND=onnx` raises instead of falling back.

On Apple Silicon, PyTorch uses the GPU (`mps`) and wins on raw inference. The ONNX path
is CPU-only and is aimed at servers, where PyTorch has no GPU to fall back on either.

## Two things that were tried and rejected

**int8 quantization.** Storing weights as 8-bit integers should shrink the model
fourfold. Measured, it did the opposite: peak memory rose to 864 MB, because dynamic
quantization dequantizes weights on every run, and accuracy collapsed — English text
scored 0.08 where the original gave 0.97, roughly inverted. DeBERTa's disentangled
attention does not survive naive dynamic quantization. Do not re-add it without
measuring both memory and accuracy.

**A single self-contained `.onnx` file.** Exporting with `dynamo=False` produces one
file with the weights embedded, which is tidier to ship. It costs 990 MB peak instead of
608 MB, because ONNX Runtime can memory-map a sidecar weights file but must read
embedded initializers into memory. The export therefore writes `model.onnx` plus
`model.onnx.data`, and both must travel together.
