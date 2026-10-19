# File migration guide

Run Python entry points with `python -m package.module` from the repository root. For example, `python run_MAPPO.py` is now `python -m training.train_traffic_shepherd`. Old root entry points have been removed; update personal launch commands using this table. Class names and checkpoint formats are unchanged.

| Previous path | Current path |
| --- | --- |
| `MAPPO.py` | [`algorithms/mappo.py`](../algorithms/mappo.py) |
| `MA2C.py` | [`algorithms/ma2c.py`](../algorithms/ma2c.py) |
| `z_merge_agent.py` | [`algorithms/z_merge.py`](../algorithms/z_merge.py) |
| `baseline.py` | [`algorithms/rule_based.py`](../algorithms/rule_based.py) |
| `on_ramp_env.py` | [`environments/pettingzoo_on_ramp.py`](../environments/pettingzoo_on_ramp.py) |
| `traffic_shepherd.py` | [`environments/traffic_shepherd.py`](../environments/traffic_shepherd.py) |
| `z_merge_env.py` | [`environments/z_merge.py`](../environments/z_merge.py) |
| `run_MAPPO.py` | [`training/train_traffic_shepherd.py`](../training/train_traffic_shepherd.py) |
| `run_flat_MARL_MAPPO.py` | [`training/train_flat_marl.py`](../training/train_flat_marl.py) |
| `run_z_merge.py` | [`training/train_z_merge.py`](../training/train_z_merge.py) |
| `measurment.py` | [`evaluation/evaluate_traffic_shepherd.py`](../evaluation/evaluate_traffic_shepherd.py) |
| `measurment_flatMARL.py` | [`evaluation/evaluate_flat_marl.py`](../evaluation/evaluate_flat_marl.py) |
| `measurment_zmerge.py` | [`evaluation/evaluate_z_merge.py`](../evaluation/evaluate_z_merge.py) |
| `run_baseline.py` | [`evaluation/evaluate_rule_based.py`](../evaluation/evaluate_rule_based.py) |
| `record.py` | [`recording/record_flat_marl.py`](../recording/record_flat_marl.py) |
| `record_shepherd.py` | [`recording/record_traffic_shepherd.py`](../recording/record_traffic_shepherd.py) |
| `data_combine.py` | [`analysis/combine_evaluation_reports.py`](../analysis/combine_evaluation_reports.py) |
| `data_combine_table.py` | [`analysis/render_results_table.py`](../analysis/render_results_table.py) |
| `chart.py` | [`analysis/plot_policy_comparisons.py`](../analysis/plot_policy_comparisons.py) |
| `poster.py` | [`analysis/plot_evaluation_metrics.py`](../analysis/plot_evaluation_metrics.py) |
| `poster_costDuration.py` | [`analysis/plot_success_vs_intervention.py`](../analysis/plot_success_vs_intervention.py) |
| `poster_speed.py` | [`analysis/plot_speed_comparison.py`](../analysis/plot_speed_comparison.py) |
| `common/plot_fn.py` | [`analysis/plot_training_rewards.py`](../analysis/plot_training_rewards.py) |
| `common/plot_benchmark_safety.py` | [`analysis/plot_safety_benchmarks.py`](../analysis/plot_safety_benchmarks.py) |
| `run_MA2C.py` | [`experiments/train_ma2c.py`](../experiments/train_ma2c.py) |
| `train_marl_rllib.py` | [`experiments/train_rllib_ppo.py`](../experiments/train_rllib_ppo.py) |
| `run_ramp_env.py` | [`experiments/run_random_on_ramp.py`](../experiments/run_random_on_ramp.py) |
| `eval_z_merge.py` | [`experiments/watch_z_merge.py`](../experiments/watch_z_merge.py) |
| `test.py` | [`experiments/hdv_control_sketch.py`](../experiments/hdv_control_sketch.py) |
| `test2.py` | [`experiments/check_rllib_imports.py`](../experiments/check_rllib_imports.py) |
| `common/Agent.py` | [`common/ma2c_agent_base.py`](../common/ma2c_agent_base.py) |
| `common/Agent_common.py` | [`common/legacy_agent_base.py`](../common/legacy_agent_base.py) |
| `common/Memory.py` | [`common/ma2c_memory.py`](../common/ma2c_memory.py) |
| `common/Memory_common.py` | [`common/mappo_memory.py`](../common/mappo_memory.py) |
| `common/Model.py` | [`common/ma2c_networks.py`](../common/ma2c_networks.py) |
| `common/Model_common.py` | [`common/mappo_networks.py`](../common/mappo_networks.py) |
| `common/utils_common.py` | [`common/legacy_tensor_utils.py`](../common/legacy_tensor_utils.py) |
| `script.ps1` | [`scripts/record_comparison.ps1`](../scripts/record_comparison.ps1) |
| `script_shell.sh` | [`scripts/resume_training.sh`](../scripts/resume_training.sh) |
| `1155282333_Justin Weng_A Human-Centric MARL.pdf` | [`reports/documents/1155282333_Justin Weng_A Human-Centric MARL.pdf`](../reports/documents/1155282333_Justin%20Weng_A%20Human-Centric%20MARL.pdf) |
| `Poster_Accel_Variance_Chart_Grouped.png` | [`reports/figures/Poster_Accel_Variance_Chart_Grouped.png`](../reports/figures/Poster_Accel_Variance_Chart_Grouped.png) |
| `Poster_Accel_Variance_Chart_WithErrors.png` | [`reports/figures/Poster_Accel_Variance_Chart_WithErrors.png`](../reports/figures/Poster_Accel_Variance_Chart_WithErrors.png) |
| `Poster_Average_Jerk_Chart_Grouped.png` | [`reports/figures/Poster_Average_Jerk_Chart_Grouped.png`](../reports/figures/Poster_Average_Jerk_Chart_Grouped.png) |
| `Poster_Average_Jerk_Chart_WithErrors.png` | [`reports/figures/Poster_Average_Jerk_Chart_WithErrors.png`](../reports/figures/Poster_Average_Jerk_Chart_WithErrors.png) |
| `Poster_Boxplot_Accel_Variance.png` | [`reports/figures/Poster_Boxplot_Accel_Variance.png`](../reports/figures/Poster_Boxplot_Accel_Variance.png) |
| `Poster_Boxplot_Average_Jerk.png` | [`reports/figures/Poster_Boxplot_Average_Jerk.png`](../reports/figures/Poster_Boxplot_Average_Jerk.png) |
| `Poster_Bubble_Plot.png` | [`reports/figures/Poster_Bubble_Plot.png`](../reports/figures/Poster_Bubble_Plot.png) |
| `Poster_Bubble_Plot_Delay.png` | [`reports/figures/Poster_Bubble_Plot_Delay.png`](../reports/figures/Poster_Bubble_Plot_Delay.png) |
| `Poster_Bubble_Tradeoff_1700vph_TopRight_Clean.png` | [`reports/figures/Poster_Bubble_Tradeoff_1700vph_TopRight_Clean.png`](../reports/figures/Poster_Bubble_Tradeoff_1700vph_TopRight_Clean.png) |
| `Poster_Bubble_Tradeoff_1700vph_with_HDV.png` | [`reports/figures/Poster_Bubble_Tradeoff_1700vph_with_HDV.png`](../reports/figures/Poster_Bubble_Tradeoff_1700vph_with_HDV.png) |
| `Poster_Bubble_Tradeoff_800vph_with_HDV.png` | [`reports/figures/Poster_Bubble_Tradeoff_800vph_with_HDV.png`](../reports/figures/Poster_Bubble_Tradeoff_800vph_with_HDV.png) |
| `Poster_Bubble_Tradeoff_Chart_1700vph.png` | [`reports/figures/Poster_Bubble_Tradeoff_Chart_1700vph.png`](../reports/figures/Poster_Bubble_Tradeoff_Chart_1700vph.png) |
| `Poster_Bubble_Tradeoff_Chart_1700vph_CleanLabels.png` | [`reports/figures/Poster_Bubble_Tradeoff_Chart_1700vph_CleanLabels.png`](../reports/figures/Poster_Bubble_Tradeoff_Chart_1700vph_CleanLabels.png) |
| `Poster_Bubble_Tradeoff_Chart_1700vph_Perfect.png` | [`reports/figures/Poster_Bubble_Tradeoff_Chart_1700vph_Perfect.png`](../reports/figures/Poster_Bubble_Tradeoff_Chart_1700vph_Perfect.png) |
| `Poster_Bubble_Tradeoff_Chart_800vph_Clean.png` | [`reports/figures/Poster_Bubble_Tradeoff_Chart_800vph_Clean.png`](../reports/figures/Poster_Bubble_Tradeoff_Chart_800vph_Clean.png) |
| `Poster_Bubble_Tradeoff_Chart_800vph_LegendFix.png` | [`reports/figures/Poster_Bubble_Tradeoff_Chart_800vph_LegendFix.png`](../reports/figures/Poster_Bubble_Tradeoff_Chart_800vph_LegendFix.png) |
| `Poster_Intervention_Duration.png` | [`reports/figures/Poster_Intervention_Duration.png`](../reports/figures/Poster_Intervention_Duration.png) |
| `Poster_Merge_Success_BarChart.png` | [`reports/figures/Poster_Merge_Success_BarChart.png`](../reports/figures/Poster_Merge_Success_BarChart.png) |
| `Poster_Radar_Chart.png` | [`reports/figures/Poster_Radar_Chart.png`](../reports/figures/Poster_Radar_Chart.png) |
| `Poster_Speed_Time_Series.png` | [`reports/figures/Poster_Speed_Time_Series.png`](../reports/figures/Poster_Speed_Time_Series.png) |
| `Poster_Speed_Time_Series_800vph.png` | [`reports/figures/Poster_Speed_Time_Series_800vph.png`](../reports/figures/Poster_Speed_Time_Series_800vph.png) |
| `Poster_Success_vs_Duration.png` | [`reports/figures/Poster_Success_vs_Duration.png`](../reports/figures/Poster_Success_vs_Duration.png) |
| `Poster_Throughput_Chart_Grouped.png` | [`reports/figures/Poster_Throughput_Chart_Grouped.png`](../reports/figures/Poster_Throughput_Chart_Grouped.png) |
| `Poster_Tradeoff_Scatter.png` | [`reports/figures/Poster_Tradeoff_Scatter.png`](../reports/figures/Poster_Tradeoff_Scatter.png) |
| `comprehensive_results_table.png` | [`reports/figures/comprehensive_results_table.png`](../reports/figures/comprehensive_results_table.png) |
| `poster_JUSTIN_WENG_CUHK.pdf` | [`reports/documents/poster_JUSTIN_WENG_CUHK.pdf`](../reports/documents/poster_JUSTIN_WENG_CUHK.pdf) |
| `poster_JUSTIN_WENG_CUHK.pptx` | [`reports/documents/poster_JUSTIN_WENG_CUHK.pptx`](../reports/documents/poster_JUSTIN_WENG_CUHK.pptx) |

Existing `results/`, `results_dongchen_baseline/`, `results_zmerge/`, `SUMO_network/`, summary datasets and historical generated-output directories retain their paths. New bytecode caches and temporary recordings are ignored; previously tracked outputs are preserved.
