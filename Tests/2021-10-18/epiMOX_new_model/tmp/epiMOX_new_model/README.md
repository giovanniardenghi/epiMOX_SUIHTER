# epiMOX
Code for simulating the course of the COVID-19 epidemic.

### Dependencies ###

The code works with python3 with the following dependencied installed

numpy pandas geopandas scipy networkx tables optimparallel matplotlib descartes pymcmcstat

### How to run ###

* To run the simulation

  ```bash
  python3 epiMOX.py TestDir
  ```

  where TestDir is the folder containing the file <input.inp> which defines the input simulation settings and the file  <params.csv> with the model parameters.
  The output is saved in the same TestDir folder.
