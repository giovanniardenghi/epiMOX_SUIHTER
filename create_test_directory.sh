#!/usr/bin/bash

printf -v today_date '%(%Y-%m-%d)T\n' -1
mkdir Tests/${today_date}

cp Tests/base/input.inp Tests/${today_date}/
cp Tests/base/param.csv Tests/${today_date}/param_est_d${today_date}-Italia.csv

