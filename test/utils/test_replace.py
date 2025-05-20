from am_combiner.utils.replace import replace_entity_name


def test_replace_entity_name():
    text_data = (
        "david is from Martinez family. His dad kept his name david martinez "
        "and he writes his name as DAVID MARTINEZ"
    )
    original_name = "David Martinez"
    replace_name = "Fake_Name"
    assert (
        replace_entity_name(text_data, original_name, replace_name)
        == "Fake_Name is from Fake_Name family. His dad kept his name "
        "Fake_Name Fake_Name and he writes his name as Fake_Name Fake_Name"
    )


def test_replace_entity_name_with_single_letter_middle_name():
    text_data = "Argentine mom hopes pope will help get son off death row."
    original_name = "John R. Rolater"
    replace_name = "Fake_Name"
    assert (
        replace_entity_name(text_data, original_name, replace_name)
        == "Argentine mom hopes pope will help get son off death row."
    )


def test_replace_entity_name_without_substring():
    text_data = "Mr. Rolater's layer Fred Johnson has been consulted."
    original_name = "John R. Rolater"
    replace_name = "Fake_Name"
    assert (
        replace_entity_name(text_data, original_name, replace_name)
        == "Mr. Fake_Name's layer Fred Johnson has been consulted."
    )


def test_replace_entity_name_full_name_replace():
    text_data = "Mr. John R. Rolater has been suspected of something."
    original_name = "John R. Rolater"
    replace_name = "Fake_Name"
    assert (
        replace_entity_name(text_data, original_name, replace_name)
        == "Mr. Fake_Name Fake_Name Fake_Name has been suspected of something."
    )


def test_replace_entity_name_full_name_replace_no_dot():
    text_data = "Mr. John R Rolater has been suspected of something."
    original_name = "John R. Rolater"
    replace_name = "Fake_Name"
    assert (
        replace_entity_name(text_data, original_name, replace_name)
        == "Mr. Fake_Name R Fake_Name has been suspected of something."
    )
