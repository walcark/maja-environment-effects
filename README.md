## Environment effects modelling within MAJA next

This repository aims to provide a simple example of the new environment effects correction proposed for MAJA next. First, we explain the main differences between the old MAJA environment effects correction and our proposition. Then, we provide an example of the new correction process.

### Comparison of the `old` and `new` environment effect correction

Following is a table with comparing both methods:

| | Old | New |
| -- | -- | -- |
| Kernel | Gaussian | King [ref] |
| Parameters | $\sigma$ | $\sigma, \gamma$ |
| Variabilité | Non ($\sigma=1,\mathrm{km}$) | Oui, le kernel dépend de AOT, RH, $\lambda$ et du mélange d'aérosol |
| Pré-traitement | Aucun | Oui, on doit calculer $P_\mathrm{5S}(\mathrm{AOT}, \mathrm{RH}, \lambda, \mathrm{Mélange d'aérosol})$ |
| Convolution | Résolution : $240m$ <br> Taille : $15\times15$ <br> Padding : miroir <br> Méthode : spatiale | Résolution : $120m$ <br> Taille : $999\times999$ <br> Padding : miroir <br> Méthode : FFT linéaire avec padding miroir | 

## Description of the new environment effect correction

### Input parameters

| Name | Dimensions | Unit | Description |
| -- | -- | -- | -- |
| `rho_toa` | `(band, y, x)` | - | Top-of-atmosphere reflectance |
| `rho_atm` | `(band, y, x)` | - | Atmospheric intrinsic reflectance |
| `tdir_down`, `tdif_down` | `(band, y, x)` | - | Direct and diffuse downward transmittances |
| `tdir_up`, `tdif_up` | `(band, y, x)` | - | Direct and diffuse upward transmittances |
| `sph_alb` | `(band, y, x)` | - | Atmospheric spherical albedo |
| `aot` | `(y, x)` | - | Aerosol optical thickness at 550 nm |
| `rh` | `(y, x)` | % | Relative humidity |
| `wl` | `(band,)` | nm | Band central wavelength |
| `species_fraction` | `(species,)` | - | Fraction of the AOT per aerosol species (CAMS), sums to 1 |
| `sigma`, `gamma` | `(aot, rh, wl)`, one file per species | km, - | King PSF parameters look-up table (`data/king/<species>.nc`) |

### Output parameters

| Name | Dimensions | Unit | Description |
| -- | -- | -- | -- |
| `psf_mean` | `(band, n, n)` | - | Mean atmospheric PSF of the scene, $n \times n$ is the image size |
| `rho_unif` | `(band, y, x)` | - | Surface reflectance under the uniform surface assumption |
| `rho_env` | `(band, y, x)` | - | Environment reflectance |
| `rho_s` | `(band, y, x)` | - | Surface reflectance corrected for environment effects |

### Pre-processing

One PSF is computed per band, for the scene-mean atmosphere $(\overline{\mathrm{AOT}}, \overline{\mathrm{RH}}, \lambda)$, on a grid of the image size ($915 \times 915$ at $120\,m$). The normalised King kernel is:

$$
\hat{K}_{\sigma,\gamma}(r) = \frac{K_{\sigma,\gamma}(r)}{\sum_{\mathrm{grid}} K_{\sigma,\gamma}},
\qquad
K_{\sigma,\gamma}(r) = \left(1 + \frac{r^2}{2\sigma^2\gamma}\right)^{-\gamma}
$$

The kernel is not linear in $(\sigma, \gamma)$, so the parameters are never interpolated: kernels are evaluated at the LUT nodes and interpolated linearly, like the species mixture:

$$
P_\mathrm{mean} = \sum_{s \in \mathrm{species}} f_s \sum_{c \in \mathcal{C}_s} w_c \, \hat{K}_{\sigma_s(c), \gamma_s(c)}
$$

where $f_s$ is the AOT fraction of species $s$, $\mathcal{C}_s$ the 8 nodes of the LUT of $s$ surrounding $(\overline{\mathrm{AOT}}, \overline{\mathrm{RH}}, \lambda)$ and $w_c$ their trilinear weights. Kernels are accumulated in place (`psf += weight * kernel`), so the peak memory is three $n \times n$ buffers whatever the number of species. See `mean_king_psf` in `src/maja_environment_effects/psf.py`.

### Atmospheric correction

For each band, with $T_\mathrm{up} = t^\mathrm{dir}_\mathrm{up} + t^\mathrm{dif}_\mathrm{up}$ and $T_\mathrm{down} = t^\mathrm{dir}_\mathrm{down} + t^\mathrm{dif}_\mathrm{down}$:

1. Uniform surface reflectance, inverting the 5S model with $\rho_s = \rho_\mathrm{env}$:

$$
\rho^{\ast} = \rho_\mathrm{toa} - \rho_\mathrm{atm},
\qquad
\rho_\mathrm{unif} = \frac{\rho^{\ast}}{S \rho^{\ast} + T_\mathrm{up} T_\mathrm{down}}
$$

2. Environment reflectance, a linear FFT convolution with mirror padding of $n/2$ pixels on every side:

$$
\rho_\mathrm{env} = \rho_\mathrm{unif} \ast P_\mathrm{mean}
$$

3. Surface reflectance:

$$
\rho_s = \frac{\rho_\mathrm{unif} \, T_\mathrm{up} \, F - \rho_\mathrm{env} \, t^{\mathrm{dif}}_{\mathrm{up}}}{t^{\mathrm{dir}}_{\mathrm{up}}},
\qquad
F = \frac{1 - S \rho_\mathrm{env}}{1 - S \rho_\mathrm{unif}}
$$

See `correct_scene` in `src/maja_environment_effects/correction.py`, and `scripts/demo.py` for an example on the synthetic scene.


