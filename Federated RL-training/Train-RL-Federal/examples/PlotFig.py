import pandas as pd
import os

from numpy.ma.core import zeros

folder_path = ("nv_center_dc_phase20/data")
output_file = "nv_center_dc_phase20/merged_output11.csv"

file_path = folder_path + ("/nv_center_time_invT2_0.0104_lr_0.001_batchsize_1024_num_steps"
                          "_1000_max_resources_1000.00_ll_True_cl_True_eval.csv")

df = pd.read_csv(file_path)
res = df['Resources']+12000
CRBmse20 = 1/(0.5*res*96+0.12/(2*2))
CRBmse10 = 1/(0.5*res*96+0.12/(1*1))
CRBmse5 = 1/(0.5*res*96+0.12/(0.5*0.5))
merged_df = pd.DataFrame({'res': res,'CRBmse5': CRBmse5,'CRBmse10': CRBmse10,'CRBmse20': CRBmse20})
merged_df.to_csv(output_file, index=False)
print(16*127,f"Merged data saved to {output_file}")
