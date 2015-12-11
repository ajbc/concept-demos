# Black Box Variational inference for Gamma-distributed latent parameters

This is the code to accompany my (guide to BBVI for gammas)[http://ajbc.io/resources/bbvi_for_gammas.pdf].

To try the demo:

```
python bbvi_simple_model.py
```

This runs 2000 iterations of BBVI and produces a `log.tsv` file.  The results an then be visualized with the included R script.

```
Rscript plot_results.R
```

Which produces `results.pdf`.  This plot shows that the expected means (found with BBVI) approach the true means (known because the data is simulated).
