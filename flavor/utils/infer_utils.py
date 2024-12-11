import json
import re


def create_id_mapping(data1, data2, id_field, match_fields):
    """
    Create a mapping of IDs between two responses based on a combination of match fields
    (e.g., bbox, file_name, etc.).
    """
    mapping = {}

    def get_match_key(item, fields):
        return tuple(str(item[field]) for field in fields if field in item)

    data1_dict = {get_match_key(item, match_fields): item for item in data1}
    data2_dict = {get_match_key(item, match_fields): item for item in data2}

    for match_value, item1 in data1_dict.items():
        if match_value in data2_dict:
            item2 = data2_dict[match_value]
            mapping[item1[id_field]] = item2[id_field]

    return mapping


def replace_ids(data, id_mapping):
    """Recursively replace IDs in the data with the mapped IDs."""
    if isinstance(data, dict):
        return {
            key: (
                replace_ids(value, id_mapping)
                if not re.match(r".*id$", key)
                else id_mapping.get(value, value)
            )  # Replace if it's an ID field
            for key, value in data.items()
        }
    elif isinstance(data, list):
        return [replace_ids(item, id_mapping) for item in data]
    else:
        return (
            id_mapping[data]
            if data in id_mapping
            else data
            if not isinstance(data, float)
            else round(data, 4)
        )


def round_floats(data, decimal_places=3):
    if isinstance(data, dict):
        return {key: round_floats(value, decimal_places) for key, value in data.items()}
    elif isinstance(data, list):
        return [round_floats(item, decimal_places) for item in data]
    elif isinstance(data, float):
        return round(data, decimal_places)
    else:
        return data


def compare_responses(response1, response2, target_fields):
    """
    Compares two API responses by creating an ID mapping and normalizing the IDs.
    Match objects using stable fields like file_name, bbox, or other non-generated fields.
    """
    id_mapping = {}

    # Create ID mappings for each section based on stable fields for matching
    for section, (id_fields, match_fields) in target_fields.items():
        for id_field in id_fields:
            if section in response1 and section in response2:
                id_mapping.update(
                    create_id_mapping(
                        response1[section], response2[section], id_field, match_fields
                    )
                )

    # Replace IDs in both responses using the mapping
    normalized_response1 = round_floats(replace_ids(response1, id_mapping))
    normalized_response2 = round_floats(replace_ids(response2, id_mapping))

    # Compare the normalized responses
    return json.dumps(normalized_response1, sort_keys=True) == json.dumps(
        normalized_response2, sort_keys=True
    )
