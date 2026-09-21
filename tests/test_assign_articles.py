"""Splitting a group's shared content between its contents entries
(find_heading_positions), on the two cases from the Živa archive where
two entries land on the same heading.

The Czech strings are test data: they are the real titles and headings.
"""
from magazine_rag.assign_articles import find_heading_positions


def block(text, type_="body"):
    return {"type": type_, "text": text}


def test_one_heading_for_two_books_stays_with_the_first_entry():
    """2017/6: one review of two books under one heading. The review's
    text must go to the first entry, the second one is absorbed."""
    blocks = [
        block("Annalisa Berta, James L. Sumich, Kit M. Kovacs: Marine "
              "Mammals. Evolutionary Biology a Felix G. Marx, Olivier "
              "Lambert, Mark D. Uhen: Cetacean Paleobiology", "heading"),
        block("Text recenze ..."),
        block("Nějaký jiný nadpis", "heading"),  # counts must disagree
        block("Ještě další", "heading"),
    ]
    entries = [{"title": "Recenze: Marine Mammals. Evolutionary Biology"},
               {"title": "Cetacean Paleobiology"}]
    assert find_heading_positions(blocks, entries) == [0, None]


def test_better_match_later_keeps_the_text_above_the_heading():
    """2020/6: the first entry matches the heading only through shared
    words and its text is above it; both keep the heading's position so
    the first gets the text before it and the second the rest."""
    blocks = [
        block("Seznam knih ..."),
        block("Ceny Nakladatelství Academia za rok 2019 a 8. ročník "
              "studentské soutěže", "heading"),
        block("Text o cenách ..."),
        block("Nějaký jiný nadpis", "heading"),  # counts must disagree
        block("Ještě další", "heading"),
    ]
    entries = [{"title": "Knihy Nakladatelství Academia"},
               {"title": "Ceny Nakladatelství Academia 2019"}]
    assert find_heading_positions(blocks, entries) == [1, 1]
