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


