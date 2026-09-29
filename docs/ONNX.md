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
pip install 'yn[onnx]' 'yn[export]'   # export needs torch too; running does not
yn export-onnx
```

The export lands in `~/.cache/yn/onnx/<model>/` and is picked up automatically.

| Variable | Effect |
|---|---|
| `YN_BACKEND` | `auto` (default: ONNX when an export exists), `torch`, or `onnx` |
| `YN_ONNX_DIR` | Where exports live. Point this at a mounted volume in a container. |
| `YN_ONNX_THREADS` | Cap ONNX Runtime's threads, so several YN processes on one small box don't each grab every core. |

`YN_BACKEND=onnx` fails loudly when there is no export. `auto` quietly falls back to
PyTorch, which is what you want on a developer machine.

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

Scores agree to four decimal places — the precision `Decision.confidence` exposes.

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
553 MB, because ONNX Runtime can memory-map a sidecar weights file but must read
embedded initializers into memory. The export therefore writes `model.onnx` plus
`model.onnx.data`, and both must travel together.
