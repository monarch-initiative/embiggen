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
    with open(path, "r") as f:
        # We parse every line of the file.
        for line in tqdm(
            f, desc="Checking given file validity", disable=not verbose):
            # Split the lines in the sublines
            splits = line.split(sep)
            # If it's the first iteration we get the expected length
            if expected_length is None:
                expected_length = len(splits)
            # Otherwise we check for consistency.
            if expected_length != len(splits):
                return False
    return True
