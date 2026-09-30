"""The facts that the four lines of a stage are given. docs/METHOD.md.

Each stage from 04 has a `four_lines()` function: predict the floor, measure
honestly, divide, double. Its `facts` argument is one of these. Each fact is
an attribute, and its name carries its unit:

    facts.bandwidth_bytes_per_s     measured on this card
    facts.weight_bytes              for example, when the stage reads weights
"""


class Facts:
    """A small bag of named numbers. Read them as attributes."""

    def __init__(self, **facts):
        self.__dict__.update(facts)

    def __repr__(self):
        items = ", ".join(f"{name}={value!r}" for name, value in self.__dict__.items())
        return f"Facts({items})"


def card_facts(**sizes):
    """-> Facts with the read bandwidth of this card, and the given sizes."""
    from cudalib.probe import read_bandwidth

    return Facts(bandwidth_bytes_per_s=read_bandwidth(), **sizes)
