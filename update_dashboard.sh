#!/bin/bash
today=$(date -Idate) # --date='Yesterday')
endforecast=$(date -Idate --date="+90 days")
echo $today

cd /home/parolini/github/epiMOX_SUIHTER

#Calibrazione LS
sed "s/TODAY/${today}/g" Tests/Dashboard/Calibration/input.inp_template > Tests/Dashboard/Calibration/input.inp
python3 epiMOX_class.py Tests/Dashboard/Calibration 
cp Tests/Dashboard/Calibration/param_est_d$today-Italia.csv Tests/Dashboard/Calibration/param_est_latest.csv

#Calibrazione MCMC
sed "s/TODAY/${today}/g" Tests/Dashboard/CalibrationRed/input.inp_templateMCMC > Tests/Dashboard/Calibration/input.inp
python3 epiMOX_MCMC_class.py Tests/Dashboard/Calibration/ 300000
python3 MCMC_postprocess.py Tests/Dashboard/Calibration/ 10000 150000

for scenario in Base Controlled Yellow Orange Red
do
sed "s/TODAY/${today}/g" Tests/Dashboard/$scenario/input.inp_template > Tests/Dashboard/$scenario/input.inp
sed -i "s/ENDFORECAST/${endforecast}/g" Tests/Dashboard/$scenario/input.inp
python3 epiMOX_class.py Tests/Dashboard/$scenario
python3 h5_to_json.py Tests/Dashboard/$scenario/simdf.h5
cp Tests/Dashboard/$scenario/simdf.json ~/dpc-covid-data/SUIHTER/$scenario.json
done

cd ~/dpc-covid-data
echo $today > SUIHTER/last_update
git pull
git add SUIHTER/*json SUIHTER/last_update
git commit -m "update ${today}"
git push
