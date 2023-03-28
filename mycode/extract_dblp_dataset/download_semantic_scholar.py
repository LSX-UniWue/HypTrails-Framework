"""
Sample usage of Semantic Scholar Academic Graph Datasets API
https://api.semanticscholar.org/api-docs/datasets
"""
from pathlib import Path
import requests
import urllib
import json

print("Start downloading semantic scholar. ")
# Get info about the latest release
latest_release = requests.get("http://api.semanticscholar.org/datasets/v1/release/latest").json()
# print(latest_release['README'])
print(latest_release['release_id'])
for dataset in latest_release['datasets']:
  print(f"Dataset: {dataset['name']}")

for name in ['papers', 'abstracts', 'citations', 'authors', 'embeddings', 'tldrs']:
  meta_file = requests.get(f'https://api.semanticscholar.org/datasets/v1/release/2023-02-21/dataset/{name}', headers={'x-api-key':'L5FJHxOv2y4y0SLYWIEMW3iNJO7o6bOcdVi7f5c5'}).json()
  print()
  with open(f"data/bibliometric_dataset/semantic_scholar/{name}_meta_file.json", "w") as f:
    json.dump(meta_file, f)

  if name == "embeddings" or name == "tldrs":
    for i, file in enumerate(meta_file['files']):
      if Path("data", "bibliometric_dataset", "semantic_scholar", f"{name}-part{i}.jsonl.gz").exists():
        print(f"Path data/bibliometric_dataset/semantic_scholar/{name}-part{i}.jsonl.gz already exists. ")
      else:
        print(f"Downloading {name} file {i}/{len(meta_file['files'])}...")
        urllib.request.urlretrieve(file, f"data/bibliometric_dataset/semantic_scholar/{name}-part{i}.jsonl.gz")

  print(f"Done with {name} ...\n")
