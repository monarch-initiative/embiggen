"""Submodule offering number to ordinal conversion for easier readability."""

special_cases = ['zeroth','first', 'second', 'third', 'fourth', 'fifth', 'sixth', 'seventh', 'eighth', 'ninth', 'tenth', 'eleventh', 'twelfth', 'thirteenth', 'fourteenth', 'fifteenth', 'sixteenth', 'seventeenth', 'eighteenth', 'nineteenth']
decades = ['twent', 'thirt', 'fort', 'fift', 'sixt', 'sevent', 'eight', 'ninet']

def number_to_ordinal(number: int) -> str:
    """Returns the string ordinal curresponding to the provided number."""
    if number < 0 or number > 99:
        raise NotImplementedError(
            "The method number_to_ordinal supported the conversion from "
            "a number to its ordinal string for values from zero upwards "
            f"to 99, but you have provided `{number}`."
        )
    if number < 20:
        return special_cases[number].capitalize()
    if number % 10 == 0:
        return f"{decades[number//10 - 2]}ieth".capitalize()
    return f"{decades[number//10 - 2].capitalize()}y{special_cases[number%10].capitalize()}"