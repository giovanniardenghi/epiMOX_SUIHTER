#!/bin/bash
today=$(date -Idate --date='Yesterday')
endforecast=$(date -Idate --date="+90 days")
echo $today

#Calibrazione LS
sed "s/TODAY/${today}/g" Tests/Dashboard/Calibration/input.inp_template > Tests/Dashboard/Calibration/input.inp
python3 epiMOX_class.py Tests/Dashboard/Calibration 
cp Tests/Dashboard/Calibration/param_est_d$today-Italia.csv Tests/Dashboard/Calibration/param_est_latest.csv

#Caso base
sed "s/TODAY/${today}/g" Tests/Dashboard/Base/input.inp_template > Tests/Dashboard/Base/input.inp
sed -i "s/ENDFORECAST/${endforecast}/g" Tests/Dashboard/Base/input.inp
python3 epiMOX_class.py Tests/Dashboard/Base

#Controlled
sed "s/TODAY/${today}/g" Tests/Dashboard/Controlled/input.inp_template > Tests/Dashboard/Controlled/input.inp
sed -i "s/ENDFORECAST/${endforecast}/g" Tests/Dashboard/Controlled/input.inp
python3 epiMOX_class.py Tests/Dashboard/Controlled

#Yellow
sed "s/TODAY/${today}/g" Tests/Dashboard/Yellow/input.inp_template > Tests/Dashboard/Yellow/input.inp
sed -i "s/ENDFORECAST/${endforecast}/g" Tests/Dashboard/Yellow/input.inp
python3 epiMOX_class.py Tests/Dashboard/Yellow

#Orange
sed "s/TODAY/${today}/g" Tests/Dashboard/Orange/input.inp_template > Tests/Dashboard/Orange/input.inp
sed -i "s/ENDFORECAST/${endforecast}/g" Tests/Dashboard/Orange/input.inp
python3 epiMOX_class.py Tests/Dashboard/Orange

#Red
sed "s/TODAY/${today}/g" Tests/Dashboard/Red/input.inp_template > Tests/Dashboard/Red/input.inp
sed -i "s/ENDFORECAST/${endforecast}/g" Tests/Dashboard/Red/input.inp
python3 epiMOX_class.py Tests/Dashboard/Red
