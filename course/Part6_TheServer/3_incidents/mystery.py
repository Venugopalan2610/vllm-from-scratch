"""Three metric functions. Each one has one fault.

DO NOT READ THIS FILE until you have written your diagnosis in
part6_inc_2_CCbreakItOnPurpose_helper.ipynb, Exercise 7.

A log is a list of dicts: arrival, first, finish, tokens, done.
"""


def tpot(log):
    # the time for each output token
    return sum(r['finish'] - r['arrival'] for r in log) / sum(r['tokens'] for r in log)


def ttft_p99(log):
    # the p99 of the time to the first token, over the requests that we can measure
    values = sorted(r['first'] - r['arrival'] for r in log if r['done'])
    return values[int(0.99 * (len(values) - 1))]


def throughput(log):
    # output tokens per second
    return sum(r['tokens'] for r in log) / sum(r['finish'] - r['arrival'] for r in log)
