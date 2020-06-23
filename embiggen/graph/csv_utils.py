from tqdm.auto import tqdm


def check_consistent_lines(path: str, sep: str, verbose: bool) -> bool:
    """Return a boolean representing if the file has consistent lines.

    Parameters
    ---------------------
    path: str,
        The path from which to be loaded.
    sep: str,
        The separators to use for the file.
    verbose: bool,
        Wethever to show loading bar or not.

    Returns
    ---------------------
    Boolean representing if file has consistent lines.
    """
    expected_length = None
    error = False
    line_splits = None
    with open(path, "r") as f:
        # We parse every line of the file.
        for i, line in tqdm(
            enumerate(f), desc="Checking given file validity", disable=not verbose):
            # Split the lines in the sublines
            line_splits = len(line.split(sep))
            # If it's the first iteration we get the expected length
            if expected_length is None:
                expected_length = line_splits
            # Otherwise we check for consistency.
            if expected_length != line_splits:
                error = True
                break
    if error:
        raise ValueError("".join((
            "Provided nodes file has malformed lines. ",
            "The provided lines have different numbers ",
            "of the given separator.\n",
            (
                "The expected number of separators was {}, "
                "but a line with {} separators was found. \n"
            ).format(expected_length, line_splits),
            "The line is the number {}.\n".format(i),
            "The given file is at path {}.\n".format(path),
            "The line in question is: '{}'\n".format(line.strip("\n")),
        )))