from am_combiner.__sanction_main__ import main_

sanction_out = main_(
    visitors=[
        "SanctionAlias",
        "SanctionBirthExtractor",
        "NationalityVisitor",
        "SanctionTermVisitor",
        "JsonSummarizer",
        "FathersNamesFromAlias",
    ],
    combiners=["ConnectedComponentsCombinerAliasK"],
    output_path="/exchange/sanctions",
    entity_types=["person"],
    cached_input=False,
    dump_visited_records=True,
    sm_types=["sanction"],
)

pep_out = main_(
    visitors=[
        "SanctionAlias",
        "SanctionBirthExtractor",
        "NationalityVisitor",
        "SanctionTermVisitor",
        "JsonSummarizer",
        "FathersNamesFromAlias",
    ],
    combiners=["ConnectedComponentsCombinerAliasK"],
    output_path="/exchange/sanctions",
    entity_types=["person"],
    cached_input=False,
    dump_visited_records=True,
    sm_types=["pep-class-1"],
)
