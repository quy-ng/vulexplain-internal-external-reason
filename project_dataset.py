from datasets import load_from_disk

def load_dataset(task):
    task_to_path = {
        "vulnerability_type": "./aspect_bigvul/dataset_vulnerability_type",
        "root_cause": "./aspect_bigvul/dataset_root_cause",
        "attack_vector": "./aspect_bigvul/dataset_attack_vector",
        "impact": "./aspect_bigvul/dataset_impact",
    }
    return load_from_disk(task_to_path[task])
