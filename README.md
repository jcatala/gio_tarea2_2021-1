# gio_tarea2_2021-1

Resolución de problema de inventario `SLC` utilizando solver en `AMPL` y `KKT (python)`.


# Python

Clonar el repositorio y correr el programa

```bash
python3 solve.py
```

Si se desea utilizar el programa en modo `verbose` (para obtener información acerca de las ecuaciones), utilizar el argumento  `verbose = True`, en la llamada a `solve()`.

Ejemplo:

```bash
➜  tarea_2 python3 solve.py
Solution for (0.99, 1.25, 250, 200, 6400, 2)
Z: 523.505977   Q:282.842712    r:534.861533
-------
Solution for (0.99, 1.25, 250, 150, 3600, 2)
Z: 433.512616   Q:244.948974    r:401.146150
-------
Solution for (0.99, 1, 250, 200, 6400, 2)
Z: 452.073831   Q:316.227766    r:534.861533
-------
Solution for (0.99, 1, 250, 150, 3600, 2)
Z: 375.646939   Q:273.861279    r:401.146150
-------
Solution for (0.95, 1.25, 250, 200, 6400, 2)
Z: 525.487770   Q:282.842712    r:536.483408
-------
Solution for (0.95, 1.25, 250, 150, 3600, 2)
Z: 435.003526   Q:244.948974    r:402.362556
-------
Solution for (0.95, 1, 250, 200, 6400, 2)
Z: 453.663102   Q:316.227766    r:536.483408
-------
Solution for (0.95, 1, 250, 150, 3600, 2)
Z: 376.842165   Q:273.861279    r:402.362556
-------
Solution for (0.99, 0.6, 250, 200, 6400, 2)
Z: 326.323494   Q:408.248290    r:534.861533
-------
Solution for (0.99, 0.6, 250, 150, 3600, 2)
Z: 273.116944   Q:353.553391    r:401.146150
-------
Solution for (0.99, 0.1, 250, 200, 6400, 2)
Z: 113.517289   Q:1000.000000   r:534.861533
-------
Solution for (0.99, 0.1, 250, 150, 3600, 2)
Z: 96.737379    Q:866.025404    r:401.146150
-------
Solution for (0.95, 0.6, 250, 200, 6400, 2)
Z: 327.281464   Q:408.248290    r:536.483408
-------
Solution for (0.95, 0.6, 250, 150, 3600, 2)
Z: 273.836944   Q:353.553391    r:402.362556
-------
Solution for (0.95, 0.1, 250, 200, 6400, 2)
Z: 113.678445   Q:1000.000000   r:536.483408
-------
Solution for (0.95, 0.1, 250, 150, 3600, 2)
Z: 96.858349    Q:866.025404    r:402.362556
-------
Solution for (0.99, 1.25, 250, 50, 400, 2)
Z: 219.092927   Q:141.421356    r:133.715383
-------
Solution for (0.99, 1.25, 250, 10, 16, 2)
Z: 87.501172    Q:63.245553     r:26.743077
-------
Solution for (0.99, 1, 250, 50, 400, 2)
Z: 191.952341   Q:158.113883    r:133.715383
-------
Solution for (0.99, 1, 250, 10, 16, 2)
Z: 77.464763    Q:70.710678     r:26.743077
-------
Solution for (0.95, 1.25, 250, 50, 400, 2)
Z: 219.594066   Q:141.421356    r:134.120852
-------
Solution for (0.95, 1.25, 250, 10, 16, 2)
Z: 87.602029    Q:63.245553     r:26.824170
-------
Solution for (0.95, 1, 250, 50, 400, 2)
Z: 192.353734   Q:158.113883    r:134.120852
-------
Solution for (0.95, 1, 250, 10, 16, 2)
Z: 77.545492    Q:70.710678     r:26.824170
-------
Solution for (0.99, 0.6, 250, 50, 400, 2)
Z: 142.760917   Q:204.124145    r:133.715383
-------
Solution for (0.99, 0.6, 250, 10, 16, 2)
Z: 58.823218    Q:91.287093     r:26.743077
-------
Solution for (0.99, 0.1, 250, 50, 400, 2)
Z: 53.375430    Q:500.000000    r:133.715383
-------
Solution for (0.99, 0.1, 250, 10, 16, 2)
Z: 23.035336    Q:223.606798    r:26.743077
-------
Solution for (0.95, 0.6, 250, 50, 400, 2)
Z: 143.002304   Q:204.124145    r:134.120852
-------
Solution for (0.95, 0.6, 250, 10, 16, 2)
Z: 58.871705    Q:91.287093     r:26.824170
-------
Solution for (0.95, 0.1, 250, 50, 400, 2)
Z: 53.415848    Q:500.000000    r:134.120852
-------
Solution for (0.95, 0.1, 250, 10, 16, 2)
Z: 23.043433    Q:223.606798    r:26.824170
-------
Compute ended...
Case (0.99, 1.25, 250, 200, 6400, 2):
         Z: 523.505977  Q: 282.842712   r: 534.861533
Case (0.99, 1.25, 250, 150, 3600, 2):
         Z: 433.512616  Q: 244.948974   r: 401.146150
Case (0.99, 1, 250, 200, 6400, 2):
         Z: 452.073831  Q: 316.227766   r: 534.861533
Case (0.99, 1, 250, 150, 3600, 2):
         Z: 375.646939  Q: 273.861279   r: 401.146150
Case (0.95, 1.25, 250, 200, 6400, 2):
         Z: 525.487770  Q: 282.842712   r: 536.483408
Case (0.95, 1.25, 250, 150, 3600, 2):
         Z: 435.003526  Q: 244.948974   r: 402.362556
Case (0.95, 1, 250, 200, 6400, 2):
         Z: 453.663102  Q: 316.227766   r: 536.483408
Case (0.95, 1, 250, 150, 3600, 2):
         Z: 376.842165  Q: 273.861279   r: 402.362556
Case (0.99, 0.6, 250, 200, 6400, 2):
         Z: 326.323494  Q: 408.248290   r: 534.861533
Case (0.99, 0.6, 250, 150, 3600, 2):
         Z: 273.116944  Q: 353.553391   r: 401.146150
Case (0.99, 0.1, 250, 200, 6400, 2):
         Z: 113.517289  Q: 1000.000000  r: 534.861533
Case (0.99, 0.1, 250, 150, 3600, 2):
         Z: 96.737379   Q: 866.025404   r: 401.146150
Case (0.95, 0.6, 250, 200, 6400, 2):
         Z: 327.281464  Q: 408.248290   r: 536.483408
Case (0.95, 0.6, 250, 150, 3600, 2):
         Z: 273.836944  Q: 353.553391   r: 402.362556
Case (0.95, 0.1, 250, 200, 6400, 2):
         Z: 113.678445  Q: 1000.000000  r: 536.483408
Case (0.95, 0.1, 250, 150, 3600, 2):
         Z: 96.858349   Q: 866.025404   r: 402.362556
Case (0.99, 1.25, 250, 50, 400, 2):
         Z: 219.092927  Q: 141.421356   r: 133.715383
Case (0.99, 1.25, 250, 10, 16, 2):
         Z: 87.501172   Q: 63.245553    r: 26.743077
Case (0.99, 1, 250, 50, 400, 2):
         Z: 191.952341  Q: 158.113883   r: 133.715383
Case (0.99, 1, 250, 10, 16, 2):
         Z: 77.464763   Q: 70.710678    r: 26.743077
Case (0.95, 1.25, 250, 50, 400, 2):
         Z: 219.594066  Q: 141.421356   r: 134.120852
Case (0.95, 1.25, 250, 10, 16, 2):
         Z: 87.602029   Q: 63.245553    r: 26.824170
Case (0.95, 1, 250, 50, 400, 2):
         Z: 192.353734  Q: 158.113883   r: 134.120852
Case (0.95, 1, 250, 10, 16, 2):
         Z: 77.545492   Q: 70.710678    r: 26.824170
Case (0.99, 0.6, 250, 50, 400, 2):
         Z: 142.760917  Q: 204.124145   r: 133.715383
Case (0.99, 0.6, 250, 10, 16, 2):
         Z: 58.823218   Q: 91.287093    r: 26.743077
Case (0.99, 0.1, 250, 50, 400, 2):
         Z: 53.375430   Q: 500.000000   r: 133.715383
Case (0.99, 0.1, 250, 10, 16, 2):
         Z: 23.035336   Q: 223.606798   r: 26.743077
Case (0.95, 0.6, 250, 50, 400, 2):
         Z: 143.002304  Q: 204.124145   r: 134.120852
Case (0.95, 0.6, 250, 10, 16, 2):
         Z: 58.871705   Q: 91.287093    r: 26.824170
Case (0.95, 0.1, 250, 50, 400, 2):
         Z: 53.415848   Q: 500.000000   r: 134.120852
Case (0.95, 0.1, 250, 10, 16, 2):
         Z: 23.043433   Q: 223.606798   r: 26.824170
The entire process took 201.746536 seconds
```