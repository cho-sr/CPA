# Public deterministic FC spectrum diagnostic

This diagnostic is limited to the existing public deterministic 4-bit FC packets and row radii. It neither loads private images/labels/raw updates nor runs a model, CPA, FIA, or image reconstruction.

The preregistered plan is `configs/public_spectrum_diagnostic_B16_T10.json`. Its detailed normalization, quantile rank mapping, fitting ranges, numerical tolerances, minimum bulk size, and failure rules are fixed before observing spectra.

For a positive-radius row, ideal decoded update divided by its quantization step is `q-7.5`. After row centering the normalized matrix is exactly `M=q-row_mean(q)` in ideal codec arithmetic. The implementation uses public codes directly and avoids an intermediate FP32 decoding roundoff. This is a spectrum of normalized code patterns; it does not reveal the true quantization error.

The covariance is `M M.T/(d-1)`. A column-chunk Gram calculation limits memory, and radius partitions reuse the corresponding principal submatrices. Every partition has its own aspect ratio `m/(d-1)`; removing the top16 eigenvalues from fitting does not change m. Original ascending rank probabilities `(j+0.5)/m` are used without remapping the retained eigenvalues to a new full CDF.

A unit-variance Marchenko–Pastur model is fit by the median observed/model quantile ratio in probabilities 0.2–0.8, with 0.1–0.9 and 0.3–0.7 sensitivity fits. Fit availability only means a numerical fit was obtained. Residuals, edge estimates and their sensitivity, raw eigenvalues, public row statistics, and feature-space principal-angle comparisons are reported without claiming a recovered source/image count.

Zero-radius rows are counted and excluded only from normalized spectrum calculations. Floor-based partitions are public sensitivity diagnostics; they do not designate rows as uninformative or remove any original observation constraint.

The independent model checks use synthetic matrices/quantiles and small CPU fixtures. The actual data run is a separate command and refuses to overwrite an existing result directory. Final input and source hashes must match those registered before the run.

The theory behind the reference model is described in [Gavish and Donoho, Optimal Shrinkage of Singular Values](https://arxiv.org/html/1405.7511). Deterministic quantization does not automatically satisfy its white-noise assumptions; [Widrow et al., Statistical Theory of Quantization](https://isl.stanford.edu/~widrow/papers/j1996statisticaltheory.pdf) describes conditions for quantization-noise models. A similar bulk shape does not prove those assumptions or demonstrate attack/defense success.
