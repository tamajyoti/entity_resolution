import pandas as pd

from am_combiner.data.data_loaders.fake_data_collection_es import summarise_name_distribution


def test_summarise_name_distribution():
    test_sample_random_names = pd.DataFrame(
        {
            "name": [
                "Anurag Vardhan",
                "Sanjeev Kumar Balyan",
                "Shintaro Nakamura",
                "Kasumi Tochinai",
                "Kazumi Miura",
                "Anatoly Mogilyov",
            ],
            "first_letter": ["a", "s", "s", "k", "k", "a"],
        }
    )
    test_names_by_first_letter = summarise_name_distribution(
        test_sample_random_names, sample_size=3
    )

    assert len(test_names_by_first_letter) == 3
    assert list(test_names_by_first_letter.first_letter.unique()) == ["a", "k", "s"]
    assert list((test_names_by_first_letter.prob.unique().round(2))) == [0.33]
