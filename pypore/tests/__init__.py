def pct_diff(a, b):
    """
    Calculates the absolute value percent difference between a and b.
    :param a: First parameter, used as the denominator.
    :param b: Second parameter, to compare.
    :return: The absolute value percent difference between the two values.
    """
    if a == 0. and b == 0.:
        return 0.
    elif a == 0.:
        return 100. * abs((a - b) / float(b))
    return 100. * abs((a - b) / float(a))