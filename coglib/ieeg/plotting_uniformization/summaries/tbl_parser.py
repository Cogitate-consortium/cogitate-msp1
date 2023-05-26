import pandas as pd
import numpy as np
from pathlib import Path

bids_root = "/mnt/beegfs/XNAT/COGITATE/ECoG/phase_2/processed/bids"
results_summary = pd.DataFrame()
# ======================================================================================================================
for roi in ["iit", "gnw"]:
    for signal in ["high_gamma", "alpha", "erp"]:
        for task in ["ti", "tr"]:
            res_file = \
                pd.read_csv(str(Path(bids_root, "derivatives/activation_analysis/sub-super/ses-V1/ieeg/results/" \
                                            "{}_{}_{}/desbadcharej_notfil_lapref/" \
                                            "sub-super_ses-V1_task-Dur_ana-activation_analysis_{}_best_lmm_results.csv")).format(signal, roi, task, roi))
            anova_res = \
                pd.read_csv(str(Path(bids_root, "derivatives/activation_analysis/sub-super/ses-V1/ieeg/results/" \
                                            "{}_{}_{}/desbadcharej_notfil_lapref/" \
                                            "sub-super_ses-V1_task-Dur_ana-activation_analysis_{}_anova_results.csv")).format(signal, roi, task, roi))
            models = ["time_win_dur_iit", "time_win_dur_gnw", "time_win_dur_cate_iit", "time_win_dur_cate_gnw"]

            mdl_res_dict = {mdl: {"p-val": [], "fstat": [], "ch": []} for mdl in models}

            # Remove any of the non relevant models:
            results = res_file.loc[res_file["model"].isin(models)]

            # Now loop through the models we have left:
            for mdl in results["model"].unique():
                mdl_res = results.loc[results["model"] == mdl]
                p_vals = []
                fstats = []
                for ch in mdl_res["group"].unique():
                    ch_res = mdl_res.loc[mdl_res["group"] == ch]
                    ch_best_anova = anova_res.loc[(anova_res["model"] == mdl) & (anova_res["group"] == ch)]
                    if "cate" in mdl:
                        if "iit" in mdl:
                            pval = ch_best_anova.loc[ch_best_anova["conditions"] == "category:iit_predictors", "p-value"]
                            fstat = ch_best_anova.loc[ch_best_anova["conditions"] == "category:iit_predictors", "F-stat"]
                        else:
                            pval = ch_best_anova.loc[ch_best_anova["conditions"] == "category:gnw_predictors", "p-value"]
                            fstat = ch_best_anova.loc[ch_best_anova["conditions"] == "category:gnw_predictors", "F-stat"]
                    else:
                        if "iit" in mdl:
                            pval = ch_best_anova.loc[ch_best_anova["conditions"] == "iit_predictors", "p-value"]
                            fstat = ch_best_anova.loc[ch_best_anova["conditions"] == "iit_predictors", "F-stat"]
                        else:
                            pval = ch_best_anova.loc[ch_best_anova["conditions"] == "gnw_predictors", "p-value"]
                            fstat = ch_best_anova.loc[ch_best_anova["conditions"] == "gnw_predictors", "F-stat"]

                    p_vals.append(pval)
                    fstats.append(fstat)
                # Get how many channels
                # Append the results to the table:
                results_summary = results_summary.append(pd.DataFrame({
                    "roi": roi,
                    "signal": signal,
                    "task": task,
                    "model": mdl,
                    "# channels": len(p_vals),
                    "Min F-stat": np.min(fstats),
                    "Max p-val": np.max(p_vals)
                }, index=[0]), ignore_index=True)
results_summary.to_csv("Results_summary.csv")

print("DONE!")
