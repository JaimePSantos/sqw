MODULARIZATION PATH CONSISTENCY UPDATE SUMMARY
==============================================

This update ensures all modular experiment files use consistent folder paths that match the static_cluster_logged_mp.py file.

FILES UPDATED:
--------------

1. generate_samples.py
   - Updated get_experiment_dir() default: base_dir="experiments_data_samples"

2. generate_probdist_from_samples.py  
   - Updated get_experiment_dir() default: base_dir="experiments_data_samples"
   - Updated find_experiment_dir_flexible() default: base_dir="experiments_data_samples"

3. generate_std_from_probdist.py
   - Updated get_experiment_dir() default: base_dir="experiments_data_samples" 
   - Updated find_experiment_dir_flexible() default: base_dir="experiments_data_samples"

4. generate_survival_probability.py
   - Updated get_experiment_dir() default: base_dir="experiments_data_samples"
   - Updated find_experiment_dir_flexible() default: base_dir="experiments_data_samples"

5. static_cluster_plot_only.py
   - Updated get_experiment_dir() default: base_dir="experiments_data_samples"
   - Updated find_experiment_dir_flexible() default: base_dir="experiments_data_samples"

6. plot_experiment_results.py
   - Updated get_experiment_dir() default: base_dir="experiments_data_samples"
   - Updated find_experiment_dir_flexible() default: base_dir="experiments_data_samples"

7. smart_loading_static.py
   - Updated find_experiment_dir_flexible() default: base_dir="experiments_data_samples"
   - Updated get_experiment_dir() default: base_dir="experiments_data_samples"
   - Updated run_experiments() default: base_dir="experiments_data_samples"
   - Updated load_experiment_results_generic() default: base_dir="experiments_data_samples"

8. smart_loading.py
   - Updated get_experiment_dir() default: base_dir="experiments_data_samples"
   - Updated run_and_save_experiment_generic() default: base_dir="experiments_data_samples"
   - Updated load_experiment_results_generic() default: base_dir="experiments_data_samples"

DIRECTORY STRUCTURE CONSISTENCY:
--------------------------------
All files now use the same default base directory structure:
- experiments_data_samples/ (for sample data)
- experiments_data_samples_probDist/ (for probability distributions)
- experiments_data_samples_std/ (for standard deviation data)  
- experiments_data_samples_survival/ (for survival probability data)

DEVIATION CONFIGURATION CONSISTENCY:
-----------------------------------
All generate_*.py files use identical deviation values:
devs = [
    (0,0),              # No noise
    (0, 0.2),           # Small noise range
    (0, 0.6),           # Medium noise range  
    (0, 0.8),           # Medium noise range  
    (0, 1),           # Medium noise range  
]

PARAMETERS CONSISTENCY:
----------------------
All files use consistent experiment parameters:
- N = 20000
- steps = N//4 = 5000
- samples = 40
- theta = math.pi/3

VERIFICATION:
------------
- No remaining function definitions with base_dir="experiments_data" (without _samples suffix)
- All function calls correctly use the configured BASE_DIR constants
- Files in Old_working_code/ directory intentionally left unchanged (legacy code)
- static_experiment_modules/ already had correct defaults
- states_to_probabilities_mp.py already had correct defaults

This ensures that all modular experiment scripts will look for and create data in the same directory structure that the static_cluster_logged_mp.py file uses, preventing path conflicts on cluster systems where existing data may be present.
