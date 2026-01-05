from datasets import load_dataset

# Login using e.g. `huggingface-cli login` to access this dataset
ds = load_dataset("hao-li/AIDev", "all_pull_request")


#save dataset to local disk
ds.save_to_disk("./AIDev_dataset_pr")

#collect all the dataset in a csv file
import pandas as pd
df = pd.DataFrame(ds['train'])
df.to_csv("AIDev_all_pull_request.csv", index=False)