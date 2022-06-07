"""Submodule offering Oxford comma list formatting."""
from typing import List

def format_list(
    words: List[str],
    bold_words: bool = False
) -> str:
    """Returns formatted list with Oxford comma.

    Parameters
    --------------------------
    words: List[str]
        The list of words to format.
    bold_words: bool = False
        Whether to use bold letters.
    """
    if len(words) == 2:
        joiner = " "
    else:
        joiner = ", "

    return joiner.join([
        "{optional_and}{open_bold}{word}{close_bold}".format(
            word=word,
            optional_and="and " if i > 0 and i == len(words) - 1 else "",
            open_bold="<b>" if bold_words else "",
            close_bold="</b>" if bold_words else "",
        )
        for i, word in enumerate(words)
    ])