# Third-party notices

## TripoSplat

This project integrates at runtime with
[VAST-AI-Research/TripoSplat](https://github.com/VAST-AI-Research/TripoSplat),
tested at commit `a78fa12d06dbf1381ca548bfac32bb68cb8c451d`.

TripoSplat is licensed under the MIT License:

```text
MIT License

Copyright (c) 2026 VAST

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

No TripoSplat source, checkpoint, sample image, or generated Gaussian asset is
vendored in this repository. `scripts/setup_upstream.sh` clones the independent
upstream repository when the user requests it.

## Runtime dependencies

PyTorch, torchvision, NumPy, Pillow, safetensors, and tqdm are runtime or
development dependencies and remain under their respective upstream licenses.
They are not vendored in this repository.
