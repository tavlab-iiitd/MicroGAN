import pandas as pd
import os
import glob

def main(root):
	origpath = os.path.join(root, "origdata", "data.csv")
	data = pd.read_csv(origpath)
	data.drop(["Unnamed: 0", "host_name", "tuberculosis", "hiv"], axis=1, inplace=True)
	data.rename(columns={'class':'label'}, inplace=True)
	labels = [0, 1, 2, 3, 4, 5]
	for i in labels:
		df = data[data['label']==i]
		df.reset_index(drop=True, inplace=True)
		df.to_csv(os.path.join(root, "origdata", f"data{i}.csv"))

allfiles = glob.glob("Chromosome_*")
for file in allfiles:
	print(f"{file}")
	main(file)
	print("-"*10)
