from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from difflib import SequenceMatcher
from typing import Dict
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.decomposition import PCA
from transformers import BertTokenizer, BertModel
import torch
from sklearn.metrics.pairwise import cosine_similarity


"""
Validation for comparing two dictionaries built on an ensemble of
comparison and similarity evaluation techniques:

-Jaccard similarity
-Direct keyword matching
-Text sequence similarity
-Principal component analysis

Similarity evaluation can be weighted for different techniques.
Weights must sum to 1.

Usage:
```python
from validation import ComprehensiveDictionaryComparator()

# Weights must sum to 1
weights = {
    "jaccard_similarity": 0.48,
    "value_similarity": 0.04, #Keyword matching
    "textual_similarity_seq": 0.0,
    "similarity_score": 0.48, #PCA
}

comparator = ComprehensiveDictionaryComparator(weights)
results = comparator.compare(input_schema, output_schema)
print(results)
```
"""


def string_similarity(str1, str2):
    return SequenceMatcher(None, str1, str2).ratio()

def compute_tfidf_similarity(values1, values2):
    vectorizer = TfidfVectorizer()
    all_values = values1 + values2
    matrix = vectorizer.fit_transform(all_values)
    sim = cosine_similarity(matrix[: len(values1)], matrix[len(values1):])
    return sim.diagonal()

def recursive_flatten(dictionary, parent_key='', sep=' '):
    items = []
    for k, v in dictionary.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(recursive_flatten(v, new_key, sep=sep).items())
        elif isinstance(v, list):
            for idx, item in enumerate(v):
                if isinstance(item, dict):
                    items.extend(recursive_flatten(item, f"{new_key}[{idx}]", sep=sep).items())
                else:
                    items.append((f"{new_key}[{idx}]", item))
        else:
            items.append((new_key, v))
    return dict(items)

def dict_similarity(input_schema, output_schema):
    input_schema = recursive_flatten(input_schema)
    output_schema = recursive_flatten(output_schema)

    keys1 = list(input_schema.keys())
    keys2 = list(output_schema.keys())
    values1 = [str(v) for v in input_schema.values()]
    values2 = [str(v) for v in output_schema.values()]

    # Jaccard Similarity based on keys
    set_keys1 = set(keys1)
    set_keys2 = set(keys2)
    intersection_keys = set_keys1 & set_keys2
    union_keys = set_keys1 | set_keys2
    jaccard_similarity = len(intersection_keys) / len(union_keys) if union_keys else 0

    # Value Exact Match Similarity for common keys
    common_keys = intersection_keys
    matching_values = sum(1 for k in common_keys if input_schema[k] == output_schema[k])
    value_similarity = matching_values / len(common_keys) if common_keys else 0

    # Textual similarity for values of common keys using SequenceMatcher
    textual_similarity_seq = sum(string_similarity(str(input_schema[k]), str(output_schema[k])) for k in common_keys) / len(common_keys) if common_keys else 0

    # Textual similarity for values using TF-IDF
    tfidf_similarities = compute_tfidf_similarity(values1, values2)
    textual_similarity_tfidf = sum(tfidf_similarities) / len(tfidf_similarities) if tfidf_similarities.size else 0

    return {
        "jaccard_similarity": jaccard_similarity,
        "value_similarity": value_similarity,
        "textual_similarity_seq": textual_similarity_seq,
        # "textual_similarity_tfidf": textual_similarity_tfidf
    }


class DictEmbedder:

    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased')
        self.model.eval()

    def embed(self, text):
        tokens = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=128, padding='max_length')
        with torch.no_grad():
            embeddings = self.model(**tokens).last_hidden_state.mean(dim=1)
        return embeddings.squeeze().numpy()

    def recursive_flatten(self, dictionary, parent_key='', sep=' '):
        items = []
        for k, v in dictionary.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self.recursive_flatten(v, new_key, sep=sep).items())
            elif isinstance(v, list):
                for idx, item in enumerate(v):
                    if isinstance(item, dict):
                        items.extend(self.recursive_flatten(item, f"{new_key}[{idx}]", sep=sep).items())
                    else:
                        items.append((f"{new_key}[{idx}]", item))
            else:
                items.append((new_key, v))
        return dict(items)

    def plot_dictionaries(self, input_schema, output_schema, names=None):
        dicts = [input_schema, output_schema]

        flat_dicts = [self.recursive_flatten(d) for d in dicts]
        embeddings = [self.embed(f"{key}: {value}") for d in flat_dicts for key, value in d.items()]

        pca = PCA(n_components=3)
        reduced = pca.fit_transform(embeddings)

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        offset = 0
        for idx, d in enumerate(flat_dicts):
            for k, v, (x, y, z) in zip(d.keys(), d.values(), reduced[offset:offset+len(d)]):
                ax.scatter(x, y, z, label=f"{names[idx] if names else f'Dict {idx + 1}'}: {k}", s=100)
                ax.text(x, y, z, f"{v}", fontsize=10)
            offset += len(d)

        ax.set_xlabel("Principal Component 1")
        ax.set_ylabel("Principal Component 2")
        ax.set_zlabel("Principal Component 3")
        ax.set_title("PCA of Dictionary Key-Value Embeddings")
        plt.show()

    def pairwise_similarity(self, input_schema, output_schema):
        dicts = [input_schema, output_schema]

        flat_dicts = [self.recursive_flatten(d) for d in dicts]
        embeddings = [self.embed(f"{key}: {value}") for d in flat_dicts for key, value in d.items()]

        similarities = cosine_similarity(embeddings)
        labels = [f"{key} (Dict {idx+1})" for idx, d in enumerate(flat_dicts) for key in d.keys()]

        return pd.DataFrame(similarities, index=labels, columns=labels)


    def heatmap(self, input_schema, output_schema):
        df = self.pairwise_similarity(input_schema, output_schema)

        flat_input_schema = self.recursive_flatten(input_schema)
        flat_output_schema = self.recursive_flatten(output_schema)

        labels1 = [f"{key} (Input Schema)" for key in flat_input_schema.keys()]
        labels2 = [f"{key} (Output Schema)" for key in flat_output_schema.keys()]

        # Extract the submatrix that represents input_schema (rows) vs output_schema (columns)
        sub_df = df.loc[labels1, labels2]

        plt.figure(figsize=(12, 10))
        sns.heatmap(sub_df, annot=True, cmap="YlGnBu", vmin=0, vmax=1)
        plt.title('Pairwise Similarity Heatmap')
        plt.show()

    def _alphabetize_dict(self, d):
        if isinstance(d, dict):
            return {k: self._alphabetize_dict(v) for k, v in sorted(d.items())}
        elif isinstance(d, list):
            return [self._alphabetize_dict(v) for v in d]
        else:
            return d

    def decision_matrix(self, input_schema, output_schema, threshold=0.8):
        input_schema = self._alphabetize_dict(input_schema)
        output_schema = self._alphabetize_dict(output_schema)
        flat_input_schema = self.recursive_flatten(input_schema)
        flat_output_schema = self.recursive_flatten(output_schema)

        keys1 = set(flat_input_schema.keys())
        keys2 = set(flat_output_schema.keys())

        missing_keys_1 = keys2 - keys1
        missing_keys_2 = keys1 - keys2

        decisions = {}

        for k in keys1:
            val1 = flat_input_schema[k]
            val2 = flat_output_schema.get(k, None)

            # if isinstance(val1, (str, int, float)) and len(str(val1)) < 50: #50 is an adjustable threshold
            #     if val1 == val2:
            #         decisions[k] = "good"
            #     else:
            #         decisions[k] = "value mismatch"
            #         print(f"Value mismatch in field '{k}': input_schema - {val1}, output_schema - {val2}")
            # else:

            sim = self.pairwise_similarity({k: val1}, {k: val2}).iloc[0, 1]
            if sim == 1:
                decisions[k] = "good"
            elif sim >= threshold:
                decisions[k] = "minor difference"
                print(f"Minor difference in field '{k}': input_schema - {val1}, output_schema - {val2}")
            else:
                decisions[k] = "value mismatch"
                print(f"Value mismatch in field '{k}': input_schema - {val1}, output_schema - {val2}")

        for k in missing_keys_1:
            decisions[k] = "missing in Dict 1"
            print(f"Field '{k}' missing in Dict 1. output_schema value: {flat_output_schema[k]}")

        for k in missing_keys_2:
            decisions[k] = "missing in Dict 2"
            print(f"Field '{k}' missing in Dict 2. input_schema value: {flat_input_schema[k]}")

        decision_df = pd.Series(decisions).to_frame('Decision')

        correct_fields = sum([1 for decision in decisions.values() if ((decision == "good") or (decision == "minor difference"))])
        total_fields = len(decisions)
        score = (correct_fields / total_fields)

        print(f"Similarity Score: {score:.2f}%")

        return score

class ComprehensiveDictionaryComparator:
    def __init__(
            self,
            weights: Dict[str, float]
        ):
        self.embedder = DictEmbedder()
        self.jaccard_similarity = weights.get('jaccard_similarity')
        self.value_similarity = weights.get('value_similarity')
        self.textual_similarity_seq = weights.get('textual_similarity_seq')
        self.similarity_score = weights.get('similarity_score')

    def compare(self, input_schema, output_schema, visualize=True, threshold=0.90):

        basic_sims = dict_similarity(input_schema, output_schema)

        if visualize:
            self.embedder.plot_dictionaries(input_schema, output_schema, names=["input_schema", "output_schema"])
            self.embedder.heatmap(input_schema, output_schema)

        decision_df = self.embedder.decision_matrix(input_schema, output_schema, threshold=threshold)

        results = {
            "basic_similarities": basic_sims,
            "decision_matrix": decision_df
        }
        expected_similarity = 1.0
        weights = {
            "jaccard_similarity": self.jaccard_similarity,
            "value_similarity": self.value_similarity,
            "textual_similarity_seq": self.textual_similarity_seq,
            # "textual_similarity_tfidf": 0.2,
            "similarity_score": self.similarity_score,
        }

        ape_values = {}
        for metric, value in results["basic_similarities"].items():
            ape = abs((expected_similarity - value) / expected_similarity)
            ape_values[metric] = ape

        # Adding the similarity score from DictEmbedder to the APE values
        similarity_score = results["decision_matrix"] / 100
        ape_values["similarity_score"] = abs((expected_similarity - similarity_score) / expected_similarity)

        weighted_mape = sum(weights[metric] * ape for metric, ape in ape_values.items())

        results["weighted_mape"] = weighted_mape

        return results
