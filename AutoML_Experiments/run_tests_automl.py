# %%
import argparse
import os
import digen
import dill as pickle

# runs an automl algorithm on the digen benchmark
# returns false if all runs for this algorithm have completed
# returns true if a new run of the algorithm was evaluated and stored. 
def run_benchmarking(experiment_dict, base_save_folder, local_cache_dir, num_runs):
    benchmark = digen.Benchmark()

    all_datasets = benchmark.dataset_names

    automl_model = experiment_dict['automl']

    print(experiment_dict['exp_name'])

    for run_i in range(0,num_runs):
        if base_save_folder is not None:
            save_folder = os.path.join(base_save_folder,"digen_results",automl_model,experiment_dict['exp_name'],f"results_run_{run_i}")
        else:
            save_folder = os.path.join(automl_model,"digen_results",experiment_dict['exp_name'],f"results_run_{run_i}")

        if os.path.exists(save_folder):
            continue
        else:
            os.makedirs(save_folder)

        print("starting this run")
        print(save_folder)

        if automl_model == "autosklearn":
            print("autosklearn")
            import autosklearn.classification
            est = autosklearn.classification.AutoSklearnClassifier(**experiment_dict['params'])

        elif automl_model == "autosklearn2":
            print("autosklearn2")
            from autosklearn.experimental.askl2 import AutoSklearn2Classifier
            import autosklearn.classification
            est = AutoSklearn2Classifier(**experiment_dict['params'])

        elif automl_model == "h2o":
            print("h20")
            from h2o.sklearn import H2OAutoMLClassifier
            import h2o
            h2o.init(   nthreads=experiment_dict['n_jobs'],
                        max_mem_size=experiment_dict['max_mem_size'])
            est = H2OAutoMLClassifier(**experiment_dict['params'])

        elif automl_model == "tpot":
            print("tpot")
            import tpot

            est = tpot.TPOTClassifier(**experiment_dict['params'])

        print("Training/evaluating original digen")
        terminate_signal_timer = experiment_dict['terminate_signal_timer']
        results = benchmark.evaluate(est, datasets=all_datasets, local_cache_dir=local_cache_dir,terminate_signal_timer=terminate_signal_timer)

        print("Done training/evaluating original digen")



        final_estimator_resutls = {}
        for dset in all_datasets:
            final_estimator_resutls[dset] = {}
            this_est = results[dset]['classifier']
            results[dset]['classifier'] = results[dset]['classifier'].__class__.__name__


            #some automl methods can't be pickled
            #extract pickleable components.
            if automl_model == "h2o":
                final_estimator_resutls[dset]['classifier'] = "h2o - load from path with h2o.load_model(model_path)"
                #final_estimator_resutls[dset]['classifier_path'] = h2o.save_model(model=this_est, path=f"{save_folder}/h2o_export", force=True)
                final_estimator_resutls[dset]['leaderboard'] = this_est.estimator.leaderboard.as_data_frame()
            elif automl_model == "tpot":
                final_estimator_resutls[dset]['classifier'] = this_est.fitted_pipeline_
                final_estimator_resutls[dset]['pareto_front_fitted_pipelines_'] = this_est.pareto_front_fitted_pipelines_
                final_estimator_resutls[dset]['evaluated_individuals_'] = this_est.evaluated_individuals_
                final_estimator_resutls[dset]['score'] = this_est._optimized_pipeline_score
                final_estimator_resutls[dset]['random_state'] = this_est.random_state
                final_estimator_resutls[dset]['params'] = this_est.get_params()
            else:
                final_estimator_resutls[dset]['classifier'] = this_est
                final_estimator_resutls[dset]['params'] = this_est.get_params()


        stmp = os.path.join(save_folder,"results.pkl")
        print(f"saving {stmp}")
        pickle.dump(results, open(os.path.join(save_folder,"results.pkl"), 'wb'))

        stmp = os.path.join(save_folder,"estimators.pkl")
        print(f"saving {stmp}")
        pickle.dump(final_estimator_resutls, open(os.path.join(save_folder,"estimators.pkl"), 'wb'))

        
        return True
    
    return False

if __name__ == '__main__':

    # Read in arguements
    parser = argparse.ArgumentParser()
    # number of threads
    parser.add_argument("-n", "--njobs", default=16,  required=False, nargs='?')
    
    # "autosklearn" when to do the AutoSklearn run
    parser.add_argument("-a", "--autosklearn", action='store_true',  required=False, nargs='?')

    #where to save the results/models
    parser.add_argument("-s", "--savepath", default=None, required=False, nargs='?')
    
    #where to store digen datasets
    parser.add_argument("-l", "--localcachedir", default=None,  required=False, nargs='?')
    
    #number of total runs for each experiment
    parser.add_argument("-r", "--num_runs", default=None, required=False, nargs='?')

    args = parser.parse_args()
    n_jobs = int(args.njobs)
    num_runs = int(args.num_runs)
    local_cache_dir = args.localcachedir
    using_autosklearn_env = args.autosklearn
    base_save_folder = args.savepath

    experiments = [

          

                    {
                    'automl': 'tpot',
                    'exp_name' : 'tpot_STC_1200s',
                    'terminate_signal_timer' : int(1200*1.2),
                    'params': {
                        'template': 'Selector-Transformer-Classifier',
                        'population_size' : 100,
                        'generations' : 100,
                        'scoring': 'roc_auc',
                        'n_jobs': n_jobs,
                        'verbosity': 2, 
                        'cv': 10, 
                        'max_time_mins': 1200/60
                    },
                    },

                    {
                    'automl': 'tpot',
                    'exp_name' : 'tpot_C_1200s',
                    'terminate_signal_timer' : int(1200*1.2),
                    'params': {
                        'template': 'Classifier',
                        'population_size' : 100,
                        'generations' : 100,
                        'scoring': 'roc_auc',
                        'n_jobs': n_jobs,
                        'verbosity': 2,
                        'cv': 10, 
                        'max_time_mins': 1200/60
                    },
                    },

                    

                    {
                    'automl': 'tpot',
                    'exp_name' : 'tpot_base_1200s',
                    'terminate_signal_timer' : int(1200*1.2),
                    'params': {
                        'template': None,
                        'population_size': 100,
                        'generations': 100,
                        'scoring': 'roc_auc',
                        'n_jobs': n_jobs,
                        'verbosity': 2,
                        'cv': 10, 
                        'max_time_mins': 1200/60
                    },
                    },

                    {
                    'automl': 'h2o',
                    'exp_name' : 'h2o_1200s',
                    'terminate_signal_timer' : int(1200*1.2),
                    'n_jobs': n_jobs,
                    'max_mem_size' : "1000G",
                    'params' : {
                        'stopping_metric': 'AUC', 
                        'sort_metric':'AUC',
                        'nfolds':10,
                        'max_runtime_secs':1200,
                        },
                    },


    ]

    if using_autosklearn_env == 'autosklearn':
        import autosklearn
        import autosklearn.classification
        import autosklearn.metrics
        autosklearn_experiments = [      



                        {
                        'automl': 'autosklearn2',
                        'exp_name' : 'autosklearn2_1200s',
                        'terminate_signal_timer' : None,
                        'params' : {'n_jobs':n_jobs, 
                                    'metric': autosklearn.metrics.roc_auc,
                                    'time_left_for_this_task': 1200,
                                    'memory_limit': 1000000
                        },
                        },


                        {
                        'automl': 'autosklearn',
                        'exp_name' : 'autosklearn_1200s',
                        'terminate_signal_timer' : None,
                        'params' : {'n_jobs':n_jobs, 
                                    'metric': autosklearn.metrics.roc_auc,
                                    'resampling_strategy_arguments': {'cv': 10},
                                    'time_left_for_this_task': 1200,
                                    'memory_limit': 1000000
                        },
                        },

                    
        ]

    if using_autosklearn_env == 'autosklearn':
        for experiment_dict in autosklearn_experiments:
            if run_benchmarking(experiment_dict, base_save_folder, local_cache_dir,num_runs=num_runs):
                break
    else:
        for experiment_dict in experiments:
            if run_benchmarking(experiment_dict, base_save_folder, local_cache_dir,num_runs=num_runs):
                break


