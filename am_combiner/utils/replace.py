import re


def replace_entity_name(text_data: str, original_name: str, replace_name: str) -> str:
    """
    Replace all original entity name to fake entity name for downstream processing.

    :param text_data:
        Article text inside which the fake entity name is to be inserted
    :param original_name:
        Original entity name that is to be replaced
    :param replace_name:
        Fake entity name which should replace the original entity
    :return:
        the transformed text data
    """
    # individual tokens replace
    name_tokens = original_name.split(" ")
    for token in name_tokens:

        if token.endswith("."):
            # Avoid token 'J.' being interpreted as 'J' + any char
            token = token.replace(".", r"\.")
            pattern = re.compile(r"\b" + token, re.IGNORECASE)
        else:
            # Create pattern ensuring token is not part of another token.
            # For example, 'john' is not part of 'johnson'
            pattern = re.compile(r"\b" + token + r"\b", re.IGNORECASE)

        text_data = pattern.sub(replace_name, text_data)

    return text_data
