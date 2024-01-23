import json

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from difflib import SequenceMatcher


def remove_escape_chars(input):
    if not input:
        return input
    # return output.encode('ascii', 'ignore').decode('unicode_escape')
    return input.replace("\\", "")


def replace_single_quotes(input_string):
    result = []
    in_string = False

    if input_string.strip().replace('\n', '')[1] == '"':
        print('No single quotes to replace')
        return input_string

    i = 0
    while i < len(input_string):
        char = input_string[i]

        if char == "'" and (i == 0 or (input_string[i-1] != "\\" and not (i > 1 and input_string[i-2:i] == "\\'"))):
            if not in_string:
                result.append('"')
                in_string = True
            else:
                result.append('"')
                in_string = False
        else:
            result.append(char)

        i += 1

    return ''.join(result)


# Fix mapping to match the expected output
def fix_key_names(dict: dict, mappings: dict, direction: str ="schema_to_json"):
    if not dict:
        return None
    
    res = {**dict}

    for key, value in mappings.items():
        if direction=="schema_to_json":
            try:
                res[value] = res.pop(key)
            except:
                continue
        elif direction=="json_to_schema":
            try:
                res[key] = res.pop(value)
            except:
                continue
                
    return res


def find_first_json_object(data_string):
    start, end, open_braces = 0, 0, 0

    for i, char in enumerate(data_string):
        if char == '{':
            if open_braces == 0:
                start = i
            open_braces += 1
        elif char == '}':
            open_braces -= 1
            if open_braces == 0:
                end = i + 1

                return data_string[start:end]
    return None


def string_similarity(str1, str2):
    return SequenceMatcher(None, str1, str2).ratio()


def compute_tfidf_similarity(values1, values2):
    vectorizer = TfidfVectorizer()
    all_values = values1 + values2
    matrix = vectorizer.fit_transform(all_values)
    sim = cosine_similarity(matrix[:len(values1)], matrix[len(values1):])
    return sim.diagonal()  # Get the diagonal elements


def dict_similarity(dict1, dict2):
    keys1 = list(dict1.keys())
    keys2 = list(dict2.keys())
    values1 = list(dict1.values())
    values2 = list(dict2.values())

    # Jaccard Similarity based on keys
    set_keys1 = set(keys1)
    set_keys2 = set(keys2)
    intersection_keys = set_keys1 & set_keys2
    union_keys = set_keys1 | set_keys2
    jaccard_similarity = len(intersection_keys) / len(union_keys) if union_keys else 0


    # Value Exact Match Similarity for common keys
    common_keys = intersection_keys
    matching_values = sum(1 for k in common_keys if dict1[k] == dict2[k])
    value_similarity = matching_values / len(common_keys) if common_keys else 0

    # Textual similarity for values of common keys using SequenceMatcher
    textual_similarity_seq = sum(string_similarity(str(dict1[k]), str(dict2[k])) for k in common_keys) / len(common_keys) if common_keys else 0

    # Textual similarity for values using TF-IDF
    tfidf_similarities = compute_tfidf_similarity(values1, values2)
    textual_similarity_tfidf = sum(tfidf_similarities) / len(tfidf_similarities) if tfidf_similarities.size else 0

    return {
        "jaccard_similarity": jaccard_similarity,
        "value_similarity": value_similarity,
        "textual_similarity_seq": textual_similarity_seq,
        "textual_similarity_tfidf": textual_similarity_tfidf
    }


def similarity_score(val1, val2):
    """Calculate similarity score of two values based on their types."""
    if isinstance(val1, str) and isinstance(val2, str):
        common_chars = sum(1 for c in val1 if c in val2)
        return 2 * common_chars / (len(val1) + len(val2))

    elif isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
        return 1 / (1 + abs(val1 - val2))

    elif isinstance(val1, list) and isinstance(val2, list):
        # For lists: average similarity of elements
        max_len = max(len(val1), len(val2))
        total_similarity = sum(similarity_score(a, b) for a, b in zip(val1, val2))
        return total_similarity / max_len

    elif isinstance(val1, dict) and isinstance(val2, dict):
        # Recursively compute similarity for nested dicts
        return dict_similarity(val1, val2)["value_similarity"]

    else:
        return 0


def dict_similarity_two(dict1, dict2):
    if not dict1 or not dict2:
        return {
            "key_similarity": 0,
            "raw_similarity": 0
        }
    # Sort the dict keys
    dict1 = {key:dict1[key] for key in sorted(dict1.keys())}
    dict2 = {key:dict2[key] for key in sorted(dict2.keys())}

    all_keys = set(dict1.keys()) | set(dict2.keys())
    total_keys = len(all_keys)

    key_similarity = len(set(dict1.keys()) & set(dict2.keys())) / total_keys
    raw_similarity = SequenceMatcher(None, json.dumps(dict1), json.dumps(dict2)).ratio()
    # value_similarity = sum(similarity_score(dict1.get(key), dict2.get(key)) for key in all_keys) / total_keys

    return {
        "key_similarity": key_similarity,
        "raw_similarity": raw_similarity
        # "value_similarity": value_similarity
    }


def parse_output(input: str) -> dict:
    try:
        predicted = json.loads(input)
        predicted = fix_key_names(predicted)
        print(predicted)
    except:
        try:
            formatted = replace_single_quotes(output)
            formatted = find_first_json_object(formatted)
            formatted = remove_escape_chars(formatted)
            formatted = json.loads(formatted)
            predicted = fix_key_names(formatted)
        except:
            predicted = None

    return predicted


def input_preprocessing(row, model_name, target_schema_str):
    if 'instruct' in model_name.lower():
        row['preprocessed_input'] = f"""
            [INST]
            Populate a JSON in the JSON_SCHEMA format from the provided TEXT_DATA.
            ---
            JSON_SCHEMA: {target_schema_str}
            ---
            TEXT_DATA: {row['input']}
            [\INST]
        """
    else:
        row['preprocessed_input'] = f"""
            Populate a JSON in the JSON_SCHEMA format from the provided TEXT_DATA.
            ---
            JSON_SCHEMA: {target_schema_str}
            ---
            TEXT_DATA: {row['input']}
        """
        
    return row