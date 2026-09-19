"""Terminal colors for the runner scripts."""

COLOR = {
    "dim": "\033[2m", "bold": "\033[1m", "green": "\033[32m",
    "yellow": "\033[33m", "cyan": "\033[36m", "red": "\033[31m",
    "reset": "\033[0m", "underline": "\033[4m",
}


def paint(text, *styles):
    return "".join(COLOR[style] for style in styles) + str(text) + COLOR["reset"]
