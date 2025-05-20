from collections import defaultdict
from typing import List, DefaultDict, Any
import spacy

from am_combiner.features.sanction import Sanction, SanctionFeatures
from am_combiner.features.common import SanctionVisitor

import deep_translator
from deep_translator import GoogleTranslator

CHAR_LEN: int = 2000


def get_sanction_dict(
    sanction: Sanction, feature_list: List[str], bypass_translation: bool
) -> DefaultDict[str, Any]:
    """Get the display fields into a default dict for any sanction."""
    display_fields = sanction.raw_entity["data"]["display_fields"]
    feature_names = [feature.name for feature in feature_list]
    sanction_dict = defaultdict(list)
    if display_fields is None:
        return sanction_dict
    for data in display_fields:
        title = data["title"].replace(" ", "_").upper()
        if title in feature_names:
            data_text = data["value"][:CHAR_LEN]
            if bypass_translation:
                sanction_dict[title].append(data_text)
                continue
            try:
                translation = GoogleTranslator(source="auto", target="en").translate(data_text)
            except deep_translator.exceptions.NotValidPayload:
                continue
            sanction_dict[title].append(translation)

    return sanction_dict


class SanctionTermVisitor(SanctionVisitor):

    """For the list of terms to be used get the terms as extracted entities."""

    def __init__(
        self,
        use_features: List[SanctionFeatures],
        bypass_translation: bool = False,
    ):
        super().__init__()
        self.out_features = use_features
        self.bypass_translation = bypass_translation

    def visit_sanction(self, sanction: Sanction) -> None:
        """Get the extracted entities for every sanction."""
        sanction_dict = get_sanction_dict(sanction, self.out_features, self.bypass_translation)
        for feature in self.out_features:
            sanction.extracted_entities[feature] = sanction.extracted_entities[feature].union(
                set(sanction_dict[feature.name])
            )


class SanctionTermSpacyVisitor(SanctionVisitor):

    """Using spacy to extract major spacy feature or org person location date."""

    def __init__(self, use_features: List[SanctionFeatures]):
        super().__init__()
        self.source_features = use_features
        self.nlp = spacy.load("en_core_web_sm")

    def visit_sanction(self, sanction: Sanction) -> None:
        """Get the spacy output for specific extracted entities for every sanction."""
        for feature in self.source_features:

            spacy_text = ", ".join(str(e) for e in sanction.extracted_entities[feature])
            spacy_tags = self.nlp(spacy_text)
            for extracted_entity in spacy_tags.ents:
                feature_ref = SanctionFeatures[extracted_entity.label_]
                feature_text = extracted_entity.text.strip()
                feature_text = feature_text.lower()
                sanction.extracted_entities[feature_ref].add(feature_text)
