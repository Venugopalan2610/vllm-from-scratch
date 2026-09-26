"""Three benchmarks of the ratio floor / time. Each one has one fault.

DO NOT READ THIS FILE until you have written your diagnosis in
part8_inc_2_CCbreakItOnPurpose_helper.ipynb, Exercise 8.

A step is a dict: computed, cached, context, ms. The card is given as
weight_bytes, params, kv_per_token, bandwidth and flops.
"""

from lab import floor_ms


def ratio_a(steps, **card):
    # the floor of the prompt: every prompt token counts
    floor = sum(floor_ms(s['computed'] + s['cached'], s['context'], **card) for s in steps)
    return floor / sum(s['ms'] for s in steps)


def ratio_b(steps, **card):
    # a decode step reads the weights: that is the floor
    floor = sum(floor_ms(s['computed'], 0, **card) for s in steps)
    return floor / sum(s['ms'] for s in steps)


def ratio_c(steps, **card):
    # the mean of the ratio of each step
    ratios = [floor_ms(s['computed'], s['context'], **card) / s['ms'] for s in steps]
    return sum(ratios) / len(ratios)
