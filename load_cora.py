import os
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split


def load_cora(verbose=0):
  # Download the dataset
  zip_file = keras.utils.get_file(
      fname="cora.tgz",
      origin="https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz",
      extract=True,
  )

  data_dir = os.path.join(os.path.dirname(zip_file), "cora")

  # Load the citations data
  citations = pd.read_csv(
      os.path.join(data_dir, "cora.cites"),
      sep="\t",
      header=None,
      names=["target", "source"],
  )
  if verbose > 0:
    print("Citations shape:", citations.shape)
    print("Sample: ")
    print(citations.sample(frac=1).head())

  column_names = ["paper_id"] + [f"term_{idx}" for idx in range(1433)] + ["subject"]
  papers = pd.read_csv(
      os.path.join(data_dir, "cora.content"), sep="\t", header=None, names=column_names,
  )
  if verbose > 0:
    print("\nPapers shape:", papers.shape)
    print("Sample, original: ")
    print(papers.sample(5).T)
    print(papers.subject.value_counts())

  class_values = sorted(papers["subject"].unique())
  class_idx = {name: id for id, name in enumerate(class_values)}
  if verbose > 0:
    print(class_values)
    print(class_idx)

  paper_idx = {name: idx for idx, name in enumerate(sorted(papers["paper_id"].unique()))}
  papers["paper_id"] = papers["paper_id"].apply(lambda name: paper_idx[name])
  citations["source"] = citations["source"].apply(lambda name: paper_idx[name])
  citations["target"] = citations["target"].apply(lambda name: paper_idx[name])
  papers["subject"] = papers["subject"].apply(lambda value: class_idx[value])
  if verbose > 0:
    print("Sample, after transformation: ")
    print(papers.sample(5).T)

  # Train test split
  train_papers, test_papers = train_test_split(papers, test_size=0.5, stratify=papers.subject, random_state=42)

  if verbose > 0:
    print(papers.shape)
    print(train_papers.shape)
    print(test_papers.shape)
    print(papers.subject.value_counts())
    print(train_papers.subject.value_counts())
    print(test_papers.subject.value_counts())

  feature_names = set(papers.columns) - {"paper_id", "subject"}

  return papers, train_papers, test_papers, paper_idx, class_idx, citations, feature_names


if __name__ == '__main__':
  papers, train_papers, test_papers, paper_idx, class_idx, citations, feature_names  = load_cora(verbose=1)

  print(citations.shape)
