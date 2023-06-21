# About

## MEG_ROI_MVPA
This folder contains code to perform the MVPA analysis on ROI based MEG source signal.
Contributors: Ling Liu

Usage:

1.0 Subject level analysis:
Use D0X_ROI_MVPA_XX.py to run subject level analysis, each code represent a analysis method.
To run the Face vs Object Category decoding analysis for Subject SA001 for experiment 1, simply use the parameter.

`python D01_ROI_MVPA_Cat.py --sub SA001 --visit V1 --cC FO`

2.0 Group level anaylsis:
D99_group_data_xx.py is used for concatenate individual subject data to one file of group data.
To concatenate Face vs Object Category decoding analysis, simply use the parameter:

`python D99_group_data_pkl.py --cC FO --analysis Cat`

3.0 Group level statistical analysis and plotting
D98_group_stat_sROI_xx.py is used for generate final results figure with the data that generated from D99_group_data_xx.py code. To generate the main Figure of Category decoding analysis, simply use the parameter:

`Python D99_group_stat_sROI_plot.py --cC FO --analysis Cat`

- config file contain the parameter used for MEG analysis
- D_MEG_function.py contain the function used on ROI_MVPA analysis
- rsa_helper_functions_meg.py revised from ieeg team rsa analysis function, used for RSA analysis
- Sublist.py/sublist_phase2.py is subject list for ROI_MVPA analysis

## Information

| | |
| --- | --- |
author_name | Ling Liu
author_affiliation | 1 School of Psychological and Cognitive Sciences, Peking University, Beijing, China; 2 School of Communication Science, Beijing Language and Culture University, Beijing, China
author_email | ling.liu@pku.edu.cn
PI_name | Huan Luo
PI_affiliation | School of Psychological and Cognitive Sciences, Peking University, Beijing, China
PI_email | huan.luo@pku.edu.cn
programming_language | python
Is a readme file included with detailed instructions for running the code? | Yes. MEG_ROI_MVPA_readme.
Is the environment file provided? | No.
Is there a config file provided to change runtime parameters? | config.py
Does the code run on the sample dataset? | no
