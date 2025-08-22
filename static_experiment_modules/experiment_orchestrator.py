"""
Main orchestrator for static noise quantum walk experiments.

This module contains the main experiment logic, coordinating all phases
of the computation from sample generation to analysis and archiving.
"""

import os
import sys
import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError
from typing import Dict, Any, List

from .experiment_config import ExperimentConfig
from .experiment_logging import (setup_master_logging, log_experiment_start, 
                                log_experiment_completion, log_phase_start, log_process_summary)
from .system_monitor import (setup_signal_handlers, print_resource_estimates, 
                            print_system_info, SHUTDOWN_REQUESTED)
from .background_executor import (start_background_process, is_background_process, 
                                 setup_background_cleanup_handlers, cleanup_background_process)
from .worker_functions import compute_dev_samples, dummy_tesselation_func
from .data_manager import (create_mean_probability_distributions_multiprocess, 
                          create_or_load_std_data, create_experiment_archive)


def run_sample_computation_phase(config: ExperimentConfig, master_logger) -> tuple:
    """Run the sample computation phase with multiprocessing."""
    log_phase_start(master_logger, "MULTIPROCESS SAMPLE COMPUTATION")
    
    # Prepare arguments for each process
    process_args = []
    for process_id, dev in enumerate(config.devs):
        args = (dev, process_id, config.N, config.steps, config.samples, 
                config.theta, config.initial_state_kwargs)
        process_args.append(args)
    
    master_logger.info(f"Launching {len(process_args)} processes...")
    
    # Track process information
    process_info = {}
    for i, (dev, process_id, *_) in enumerate(process_args):
        dev_str = f"{dev}" if isinstance(dev, (int, float)) else f"{dev[0]}_{dev[1]}"
        log_file = os.path.join(config.process_log_dir, f"process_dev_{dev_str}_pid_{process_id}.log")
        process_info[dev] = {
            "process_id": process_id,
            "log_file": log_file,
            "start_time": None,
            "end_time": None,
            "status": "pending"
        }
    
    # Execute processes concurrently with robust error handling
    max_retries = 3
    retry_delay = 30  # seconds
    completed_samples = 0
    total_samples = len(config.devs) * config.samples
    process_results = []
    
    for attempt in range(max_retries):
        if SHUTDOWN_REQUESTED:
            master_logger.warning("Shutdown requested, stopping sample computation")
            break
            
        master_logger.info(f"Sample computation attempt {attempt + 1}/{max_retries}")
        
        try:
            with ProcessPoolExecutor(max_workers=config.max_processes) as executor:
                # Track process start times
                for dev in config.devs:
                    process_info[dev]["start_time"] = time.time()
                    process_info[dev]["status"] = "running"
                
                # Submit all tasks
                future_to_dev = {executor.submit(compute_dev_samples, args): args[0] 
                               for args in process_args}
                
                # Process completed tasks with timeout
                for future in as_completed(future_to_dev, timeout=config.process_timeout):
                    if SHUTDOWN_REQUESTED:
                        master_logger.warning("Shutdown requested during process execution")
                        break
                        
                    dev = future_to_dev[future]
                    try:
                        result = future.result()
                        process_results.append(result)
                        
                        # Update process info
                        process_info[dev]["end_time"] = time.time()
                        process_info[dev]["status"] = "completed" if result["success"] else "failed"
                        
                        if result["success"]:
                            completed_samples += result["computed_samples"]
                            master_logger.info(f"Process {result['process_id']} completed dev {dev}: "
                                             f"{result['computed_samples']} samples in {result['total_time']:.1f}s")
                        else:
                            master_logger.error(f"Process {result['process_id']} failed for dev {dev}: {result['error']}")
                            
                    except Exception as e:
                        master_logger.error(f"Exception in process for dev {dev}: {str(e)}")
                        process_info[dev]["status"] = "error"
                        process_results.append({
                            "dev": dev,
                            "process_id": process_info[dev]["process_id"],
                            "computed_samples": 0,
                            "total_time": 0,
                            "log_file": process_info[dev]["log_file"],
                            "success": False,
                            "error": str(e)
                        })
                
                # If we got here without major errors, break the retry loop
                break
                
        except TimeoutError:
            master_logger.error(f"Timeout occurred in sample computation (attempt {attempt + 1})")
            if attempt < max_retries - 1:
                master_logger.info(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                master_logger.error("Max retries exceeded for sample computation")
        except Exception as e:
            master_logger.error(f"Critical error in sample computation attempt {attempt + 1}: {str(e)}")
            if attempt < max_retries - 1:
                master_logger.info(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                master_logger.error("Max retries exceeded due to critical errors")
                raise
    
    return process_results, completed_samples, total_samples, process_info


def run_analysis_phase(config: ExperimentConfig, master_logger):
    """Run the analysis phase including mean probability and std calculations."""
    log_phase_start(master_logger, "ANALYSIS - MEAN PROBABILITY DISTRIBUTIONS")
    
    try:
        # Smart load or create mean probability distributions
        from smart_loading_static import (
            check_mean_probability_distributions_exist, 
            load_mean_probability_distributions
        )
        
        if check_mean_probability_distributions_exist(dummy_tesselation_func, config.N, config.steps, 
                                                    config.devs, config.samples,
                                                    "experiments_data_samples_probDist", "static_noise", config.theta):
            master_logger.info("Loading existing mean probability distributions...")
            mean_results = load_mean_probability_distributions(dummy_tesselation_func, config.N, config.steps, 
                                                             config.devs, config.samples,
                                                             "experiments_data_samples_probDist", "static_noise", config.theta)
        else:
            master_logger.info("Creating mean probability distributions with multiprocessing...")
            mean_prob_results = create_mean_probability_distributions_multiprocess(
                dummy_tesselation_func, config.N, config.steps, config.devs, config.samples,
                "experiments_data_samples", "experiments_data_samples_probDist", "static_noise", config.theta,
                config.use_multiprocess_mean_prob, config.max_mean_prob_processes, master_logger
            )
            
            # Load the created mean probability distributions
            mean_results = load_mean_probability_distributions(dummy_tesselation_func, config.N, config.steps, 
                                                             config.devs, config.samples,
                                                             "experiments_data_samples_probDist", "static_noise", config.theta)
        
    except Exception as e:
        error_msg = f"Warning: Could not smart load/create mean probability distributions: {e}"
        print(f"[WARNING] {error_msg}")
        master_logger.error(error_msg)
        import traceback
        master_logger.error(traceback.format_exc())
        mean_results = None

    # Create or load standard deviation data
    try:
        stds = create_or_load_std_data(
            mean_results, config.devs, config.N, config.steps, config.samples, dummy_tesselation_func,
            "experiments_data_samples_std", "static_noise", theta=config.theta
        )
        
        # Print final std values for verification
        for i, (dev, std_values) in enumerate(zip(config.devs, stds)):
            if std_values:
                final_std = std_values[-1] if len(std_values) > 0 else 0
                master_logger.info(f"Dev {dev}: Final standard deviation = {final_std:.6f}")
                
    except Exception as e:
        print(f"[WARNING] Warning: Could not create/load standard deviation data: {e}")
        stds = []
    
    return mean_results, stds


def run_plotting_phase(config: ExperimentConfig, stds: List, master_logger):
    """Run the plotting phase if enabled."""
    if not config.enable_plotting:
        print("\n[PLOT] Plotting disabled (enable_plotting=False)")
        return
    
    # Plot standard deviation vs time
    print("\n[PLOT] Creating standard deviation vs time plot...")
    try:
        from jaime_scripts import plot_std_vs_time_qwak
        
        if config.use_loglog_plot:
            plot_filename = f"static_noise_std_vs_time_loglog_N{config.N}_samples{config.samples}.png"
            plot_std_vs_time_qwak(stds, config.devs, config.N, config.samples, 
                                 use_loglog=True, save_figure=config.save_figures, 
                                 filename=plot_filename if config.save_figures else None)
        else:
            plot_std_vs_time_qwak(stds, config.devs, config.N, config.samples, 
                                 use_loglog=False, save_figure=config.save_figures)
        
        master_logger.info("Standard deviation plot created successfully")
    except Exception as e:
        error_msg = f"Error creating standard deviation plot: {e}"
        print(f"[ERROR] {error_msg}")
        master_logger.error(error_msg)

    # Plot final probability distributions if enabled
    if config.plot_final_probdist:
        print("\n[PLOT] Creating final probability distribution plot...")
        try:
            from jaime_scripts import plot_final_probability_distributions
            
            plot_final_probability_distributions(config.devs, config.N, config.steps, config.samples,
                                                config.theta, save_figure=config.save_figures)
            master_logger.info("Final probability distribution plot created successfully")
        except Exception as e:
            error_msg = f"Error creating final probability distribution plot: {e}"
            print(f"[ERROR] {error_msg}")
            master_logger.error(error_msg)
    else:
        print("\n[PLOT] Final probability distribution plotting disabled (plot_final_probdist=False)")


def run_archiving_phase(config: ExperimentConfig, master_logger):
    """Run the archiving phase if enabled."""
    if not config.create_tar_archive:
        print("[INFO] Archiving disabled (create_tar_archive=False)")
        return None
    
    archive_name = create_experiment_archive(config.N, config.samples, config.use_multiprocess_archiving, 
                                           config.max_archive_processes, config.exclude_samples_from_archive, master_logger)
    if archive_name:
        print(f"[OK] Archive created: {archive_name}")
        master_logger.info(f"Archive created successfully: {archive_name}")
    else:
        print("[WARNING] Archive creation failed or skipped")
        master_logger.warning("Archive creation failed or skipped")
    
    return archive_name


def run_static_experiment(config: ExperimentConfig) -> Dict[str, Any]:
    """
    Run the static noise quantum walk experiment with the given configuration.
    
    Args:
        config: ExperimentConfig object containing all experiment parameters
        
    Returns:
        Dictionary containing experiment results and metadata
    """
    
    # Setup signal handlers
    setup_signal_handlers()
    
    # Print configuration summary
    config.print_config_summary()
    
    # Print resource estimates
    estimates = config.get_resource_estimates()
    print_resource_estimates(estimates)
    print_system_info()
    
    # Handle background execution
    if config.run_in_background and not is_background_process():
        script_path = os.path.abspath(__file__)
        success = start_background_process(script_path, config.background_log_file, config.background_pid_file)
        if success:
            return {"mode": "background_started", "background_pid_file": config.background_pid_file}
        else:
            print("Failed to start background process, continuing in foreground...")
    
    # Setup background cleanup if we're the background process
    if is_background_process():
        setup_background_cleanup_handlers(config.background_log_file, config.background_pid_file)
        print("Running in SAFE background mode...")
        print(f"   Process ID: {os.getpid()}")
        print(f"   Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Import required modules
    try:
        from sqw.experiments_expanded_static import running
        from sqw.states import uniform_initial_state, amp2prob
        from jaime_scripts import prob_distributions2std, plot_std_vs_time_qwak
        from smart_loading_static import smart_load_or_create_experiment, get_experiment_dir
        
        print("Successfully imported all required modules")
    except ImportError as e:
        error_msg = f"Error: Could not import required modules: {e}"
        print(error_msg)
        
        if is_background_process():
            try:
                with open(config.background_log_file, 'a') as f:
                    f.write(f"\n{error_msg}\n")
                cleanup_background_process(config.background_pid_file)
            except:
                pass
        
        raise

    print("Starting static noise quantum walk experiment...")
    
    # Setup master logging
    master_logger, master_log_file = setup_master_logging()
    log_experiment_start(master_logger, config)
    
    # Start timing the main experiment
    start_time = time.time()
    
    # Sample computation phase
    experiment_time = 0
    process_results = []
    completed_samples = 0
    total_samples = len(config.devs) * config.samples
    
    if not config.skip_sample_computation:
        sample_start_time = time.time()
        process_results, completed_samples, total_samples, process_info = run_sample_computation_phase(config, master_logger)
        experiment_time = time.time() - sample_start_time
        
        # Log final results
        log_phase_start(master_logger, "MULTIPROCESS COMPUTATION COMPLETED")
        master_logger.info(f"Total execution time: {experiment_time:.2f} seconds")
        master_logger.info(f"Total samples computed: {completed_samples}/{total_samples}")
        
        # Log process summary
        log_process_summary(master_logger, process_results, process_info)
        
        print(f"\n[COMPLETED] Multiprocess sample computation completed in {experiment_time:.2f} seconds")
        print(f"Total samples computed: {completed_samples}")
        successful_processes = len([r for r in process_results if r["success"]])
        print(f"Successful processes: {successful_processes}/{len(config.devs)}")
        print(f"Master log file: {master_log_file}")
        print(f"Process log directory: {config.process_log_dir}")
        
    else:
        master_logger.info("SKIPPING SAMPLE COMPUTATION")
        print("=== SKIPPING SAMPLE COMPUTATION ===")
        print("Sample computation disabled - proceeding to analysis phase")

    # Early exit if only computing samples
    if config.calculate_samples_only:
        master_logger.info("SAMPLES ONLY MODE - ANALYSIS SKIPPED")
        print("\n=== SAMPLES ONLY MODE - ANALYSIS SKIPPED ===")
        print("Sample computation completed. Skipping analysis and plotting.")
        
        # Create tar archive if enabled (even in samples-only mode)
        archive_name = run_archiving_phase(config, master_logger)
        
        print("To run analysis on existing samples, set:")
        print("  calculate_samples_only = False")
        print("  skip_sample_computation = True")
        
        total_time = time.time() - start_time
        log_experiment_completion(master_logger, total_time, "samples_only")
        
        return {
            "mode": "samples_only",
            "config": config,
            "total_time": total_time,
            "completed_samples": completed_samples,
            "process_results": process_results,
            "master_log_file": master_log_file,
            "archive_name": archive_name
        }

    # Analysis phase
    print("\n=== ANALYSIS PHASE ===")
    print("Loading existing samples and computing analysis...")
    
    mean_results, stds = run_analysis_phase(config, master_logger)
    
    # Plotting phase
    run_plotting_phase(config, stds, master_logger)
    
    # Archiving phase
    archive_name = run_archiving_phase(config, master_logger)

    print("Static noise experiment completed successfully!")
    total_time = time.time() - start_time
    log_experiment_completion(master_logger, total_time, config.execution_mode)
    print(f"Total execution time: {total_time:.2f} seconds")
    
    # Print performance summary
    _print_performance_summary(config, completed_samples, process_results, experiment_time, 
                              master_log_file, archive_name)
    
    return {
        "mode": "full_pipeline",
        "config": config,
        "total_time": total_time,
        "completed_samples": completed_samples,
        "process_results": process_results,
        "master_log_file": master_log_file,
        "archive_name": archive_name,
        "mean_results": mean_results,
        "stds": stds
    }


def _print_performance_summary(config: ExperimentConfig, completed_samples: int, 
                              process_results: List, experiment_time: float,
                              master_log_file: str, archive_name: str):
    """Print detailed performance summary."""
    print("\n=== Performance Summary ===")
    print(f"Execution mode: {config.execution_mode}")
    print(f"System size (N): {config.N}")
    print(f"Time steps: {config.steps}")
    print(f"Samples per deviation: {config.samples}")
    print(f"Number of deviations: {len(config.devs)}")
    print(f"Multiprocessing: {config.max_processes} max processes")
    
    if not config.skip_sample_computation:
        print(f"Total quantum walks computed: {completed_samples}")
        successful_processes = len([r for r in process_results if r["success"]])
        print(f"Successful processes: {successful_processes}/{len(config.devs)}")
        if experiment_time > 0 and completed_samples > 0:
            avg_time_per_qw = experiment_time / completed_samples
            print(f"Average time per quantum walk: {avg_time_per_qw:.3f} seconds")
    else:
        print(f"Expected quantum walks: {len(config.devs) * config.samples} (sample computation skipped)")
    
    print("\n=== Multiprocessing Log Files ===")
    print(f"Master log: {master_log_file}")
    print(f"Process logs directory: {config.process_log_dir}")
    if process_results:
        print("Individual process logs:")
        for result in process_results:
            status = "SUCCESS" if result["success"] else "FAILED"
            print(f"  {status}: Dev {result['dev']} - {result['log_file']}")

    print("\n=== Configuration Details ===")
    print("Available execution modes:")
    print("1. Full Pipeline (default): Compute samples + analysis + plots + archive")
    print("2. Samples Only: Set calculate_samples_only = True")
    print("3. Analysis Only: Set skip_sample_computation = True")
    print("4. Custom: Adjust individual toggles for plotting, archiving, etc.")
    
    print("\n=== Static Noise Details ===")
    print("Static noise model:")
    print(f"- dev=0: Perfect static evolution with theta={config.theta:.3f} (no noise)")
    print("- dev>0: Random deviation applied to Hamiltonian edges with range 'dev'")
    print("- Each sample generates different random noise for edge parameters")
    print("- Mean probability distributions average over all samples")
    print("- Tessellations are built-in (alpha and beta patterns)")
    print("- MULTIPROCESSING: Each deviation value runs in separate process")
