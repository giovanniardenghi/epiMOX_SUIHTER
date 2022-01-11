#!/bin/bash
today=$(date -Idate)
endforecast=$(date -Idate --date="+90 days")
echo $today

cd /home/parolini/github/epiMOX_SUIHTER

casedir=Tests/DashboardMCMC/

#Calibrazione LS
sed "s/TODAY/${today}/g" $casedir/input.inp_template > $casedir/input.inp
python3 epiMOX_class.py $casedir 
cp $casedir/param_est_d$today-Italia.csv $casedir/param_est_latest.csv

#Calibrazione MCMC
rm -rf $casedir/MCMC
mkdir -p $casedir/MCMC_forecasts_$today
sed "s/TODAY/${today}/g" $casedir/input.inp_templateMCMC > $casedir/input.inp
python3 epiMOX_MCMC_class.py $casedir/ 300000
sed -i "s/Tf = ${today}/Tf = ${endforecast}/g" $casedir/input.inp
python3 MCMC_postprocess_dashboard.py $casedir/ 1000 150000
for quantile in 5 025 975 
do
mv $casedir/simdf_MCMC_$quantile.json $casedir/MCMC_forecasts_$today/
done

for scenario in Controlled Yellow Orange Red
do
echo $scenario
sed "s/TODAY/${today}/g" $casedir/input.inp_templateMCMCscenario > $casedir/input.inp
sed -i "s/ENDFORECAST/${endforecast}/g" $casedir/input.inp
sed -i "s/SCENARIO/${scenario}/g" $casedir/input.inp
python3 MCMC_postprocess_dashboard.py $casedir/ 1000 150000
for quantile in 5 025 975 
do
mv $casedir/simdf_MCMC_$quantile.json $casedir/MCMC_forecasts_$today/${scenario}_${quantile}.json
done
done

cp -r $casedir/MCMC_forecasts_$today/*json ~/dpc-covid-data/SUIHTER/MCMC_forecasts/

cd ~/dpc-covid-data
echo $today > SUIHTER/last_update
git pull
git add SUIHTER/MCMC_forecasts/*json SUIHTER/last_update
git commit -m "update ${today}"
git push
