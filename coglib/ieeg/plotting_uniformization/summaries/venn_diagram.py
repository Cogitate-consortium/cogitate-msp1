from matplotlib_venn import venn3, venn3_circles
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
new_rc_params = {'text.usetex': False,
                 "svg.fonttype": "none"
                 }
mpl.rcParams.update(new_rc_params)

overlap = pd.read_csv(r"C:\Users\alexander.lepauvre\Documents\PhD\COGITATE\activation_venn_overlap.csv")

model_channels = overlap["modelling"].dropna().to_list()
decoding_channels = overlap["duration_decoding"].dropna().to_list()
tracking_channels = overlap["duration_tracking"].dropna().to_list()

set1 = set(model_channels)
set2 = set(decoding_channels)
set3 = set(tracking_channels)
venn3([set1, set2, set3], ('Models', 'Decoding', 'Tracking'))
plt.savefig(r"C:\Users\alexander.lepauvre\Documents\PhD\COGITATE\analyses_overlap_venn.svg")

