Each call of run_test_automl.py will run the next algorithm in the experiments list that does not yet have results. Each script calls benchmark.evaluate on only one algorithm one time and then exits. Script can also be executed multiple times in parallel. 

The two .sh files includes the settings for the Slurm scheduler as well as flags passed into run_test_automl.py .


## Environments
Using the latest master branch on the TPOT repository https://github.com/epistasislab/tpot
Using a modified DIGEN fork (found in this repository), to add in a termination signal.

Environments were set up as follows: 

```
conda create --name tpot_digen_env_final -c h2oai -c plotly -c conda-forge xgboost dask dask-ml scikit-mdr skrebate dill jupyter seaborn optuna pandas scipy seaborn matplotlib plotly h2o=3.38.0.1
conda activate tpot_digen_env_final
pip install -e tpot
pip install -e digen
pip install -U kaleido
```


```
conda create --name autosklearn_digen_env_final python
pip install auto-sklearn=0.14.5
pip install -e digen
conda install -c conda-forge dill jupyter plotly pynisher=0.6.4
conda activate autosklearn_digen_env_final
```