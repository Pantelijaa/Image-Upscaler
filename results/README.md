# results/

Written by every run of the benchmark (`CMakeTarget`). Only the screenshots used in
the main README are tracked; the rest is ignored by git.

```
results/
├── results.csv                    # One row per (dataset, scale, method, image): psnr, ssim, time_ms
├── <Set>/<Method>/x<scale>/       # Every upscaled image, e.g. Set5/vdsr_x2_v2_int8/x2/baby.png
├── Averages.png                   # Screenshot of the averages table printed at the end of a run
└── CMakeTargets.png               # Screenshot of the CMake Targets View in Visual Studio
```

`<Method>` is `Bilinear`, `Bicubic` or the ONNX file name without extension.
