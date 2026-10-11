# Based on seminar materials

# Don't forget to support cases when target_text == ''

from collections.abc import Sequence

import editdistance


def calc_cer(target_text: str, predicted_text: str) -> float:
    return editdistance.eval(target_text, predicted_text) / max(1, len(target_text))


def calc_wer(target_text: str, predicted_text: str) -> float:
    target_words = target_text.split()
    return editdistance.eval(target_words, predicted_text.split()) / max(
        1, len(target_words)
    )


# Own realisation from seminar, not optimal at all, so I use editdistance
def edit_distance(target_text: Sequence[str], prod_text: Sequence[str]) -> int:
    dp = [[0 for j in range(len(prod_text) + 1)] for i in range(len(target_text) + 1)]

    for i in range(1, len(target_text) + 1):
        dp[i][0] = dp[i - 1][0] + 1

    for j in range(1, len(prod_text) + 1):
        dp[0][j] = dp[0][j - 1] + 1

    for i in range(1, len(target_text) + 1):
        for j in range(1, len(prod_text) + 1):
            dp[i][j] = min(
                dp[i][j - 1] + 1,  # Insertion
                dp[i - 1][j] + 1,  # Deletion
                dp[i - 1][j - 1]
                + int(
                    target_text[i - 1] != prod_text[j - 1]
                ),  # Correct and substitution
            )

    return dp[-1][-1]
