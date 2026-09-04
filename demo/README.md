# Research demo

This is a dependency-free static page for the paper's argument, quantitative
evidence, and curated listening cases.

```bash
python -m http.server 8000 --directory demo
```

Open `http://127.0.0.1:8000/`.

The result selector has three independent states:

1. Clean TEST;
2. frozen Noisy DEV selection/ablation evidence;
3. frozen Noisy TEST, initially pending.

When a validated TEST package becomes available, `frozen-noisy-test.js` is
generated from the frozen report. `app.js` assigns it only to `noisyTest`, so
the DEV table cannot be replaced by the TEST import.
