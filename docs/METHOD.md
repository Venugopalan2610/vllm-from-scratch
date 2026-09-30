# The method

This course builds an engine. The engine stays in this repository. The method
is what you take with you.

A system stops at its last stage. A method goes with you to the next engine:
SGLang, TensorRT-LLM, the code at your job, a machine that does not exist
yet. This page is the whole method. Each notebook and each stage uses a part
of it, and the notebooks label each use.

---

## Before anything: what is given, and what is asked

Write two lists before you start any stage, exercise or ticket:

- **Given:** everything that you have before you start. The code, the inputs,
  the results of earlier stages, the facts and the numbers.
- **Asked:** the one thing that you must produce, and how you will know that
  it is correct.

**Why:** when the given is completely clear, the asked becomes easy to
understand. Most confusion is a given that you did not see. A question that
looks hard is often a short step from a fact that you already have.

**When:** at the start of every stage, every exercise and every ticket.

**In the course:** `./vc guide` shows WHAT IS GIVEN before WHAT IS ASKED.
Each stage lists its given in `stages.yaml`. An incident ticket gives you a
symptom and evidence, and asks for four lines: the cause, the number that
proves it, the fix and the guard.

---

## When your mind is empty, ask this

> **What is the least that this can possibly cost?**

This question always has an answer. It is the first thing to ask about what
is given. Count three things:

- the **bytes** that the work must move from memory,
- the **arithmetic** that it must do,
- the **launches** (calls from the CPU to the GPU) that it must make.

Divide each count by the speed of the machine for it. The largest result is
the floor. No code can be faster than the floor. Now you have a number to
compare with, and every measurement after that means something.

This is the first prediction, every time. The six moves below follow from it.

---

## The six moves

Each move answers one question. For each move, this page tells you **when** to
use it and **why**, not only what it is.

### 1. Predict

- **Ask:** what number do I expect, before I run anything?
- **When:** before every measurement. No exceptions.
- **Why:** a measurement with no prediction cannot surprise you, and a
  surprise is the only thing that teaches you. When you predict first, the
  gap between the prediction and the measurement tells you what you did not
  count.
- **Simple tool:** paper. Write the number and its units.
- **Trap:** you "predict" after you see the result. Write the number down
  first, in a cell, or with `./vc note`.
- **In the course:** `part0_gpu_3_CCpredictThenMeasure`, `part4_adm_2`.

### 2. Find what waits

- **Ask:** is this waiting for bytes, for arithmetic, for launches, or for the
  scheduler?
- **When:** something is slow, and you want to make it faster.
- **Why:** each type of wait has a different fix. Less arithmetic does not
  help code that waits for bytes. A faster kernel does not help code that
  waits for Python.
- **Simple tool:** the floor. Compare the time with the time of the bytes, of
  the arithmetic, and of the launches.
- **Formal tool:** the roofline. The ridge tells you which side you are on.
- **Trap:** you assume that the slow part is the part with the most code.
- **In the course:** `part1_arith_1` (the ridge), `part5_cg_1` (the Python
  tax), `part2_pad_2` ("None of this is a hardware problem").

### 3. Divide

- **Ask:** what is this number divided by the floor?
- **When:** you want to compare two results: two versions, two cards, two
  students.
- **Why:** a raw number, such as 70 tokens per second, is true on one machine
  only. A ratio to the floor cancels the machine. It stays true on every card.
- **Simple tool:** one division. Check that the units cancel.
- **Trap:** you compare milliseconds from two different cards.
- **In the course:** `part1_arith_1`, `part8_roof_1` ("The ratio that
  travels"), `part8_eng_2`, stage 28.

### 4. Read the shape

- **Ask:** how does the cost change when the input grows?
- **When:** something grows: the context, the batch, the number of users.
- **Why:** a number is true at one size. The shape tells you what happens at
  the next size, which is where production breaks.
- **Simple tool:** **double the input, and divide.** `t(2L) / t(L)`:
  about 2 is linear, about 4 is quadratic, about 1 is flat.
- **Formal tool:** a log-log plot, where the slope is the exponent. Then a fit,
  such as `np.polyfit`, to separate the terms.
- **Trap:** you read a table at small sizes, where every curve looks like a
  line.
- **In the course:** `part0_meas_2`, `part1_kv_1`, `part2_cont_1` ("The shape
  of the static curve is the bug"), `part6_met_1` ("The shape to notice is the
  collapse").

### 5. Find the bill

- **Ask:** this fix moved a cost. Where did the cost go?
- **When:** a change makes something faster or smaller, and especially when
  it looks free.
- **Why:** nothing is free. A fix usually moves a cost to a place that you do
  not measure yet. That place is the next problem, and often the next stage.
- **Simple tool:** list what the fix reads, writes and waits for, before and
  after.
- **Trap:** you measure only the thing that you tried to improve.
- **In the course:** `part3_blk_2` ("Nothing is free"), `part3_gat_2` ("That
  number is the bill for stage 06"), `part5_cg_2`. Each stage has a `because:`
  line in `stages.yaml`: the bill of the stage before it.

### 6. Keep an oracle

- **Ask:** how do I prove that the output did not change?
- **When:** you change how data is stored or computed: a cache, a kernel, a
  number format, a batch layout.
- **Why:** in an engine, most bugs do not raise an error. They give fluent,
  correct-looking text that is wrong. Only a comparison with a trusted
  version finds them.
- **Simple tool:** keep the slow, correct version. Compare token for token.
- **Formal tool:** a distance between two probability distributions, such as
  the KL divergence, when the outputs are not exactly equal.
- **Trap:** you check that the output "looks right".
- **In the course:** stage 07 (the oracle), `part7_spc_1`, `part7_qnt_1`,
  `part8_gr_1` ("A zero is not a harmless value").

---

## What to predict: the triggers

An expert does not invent a question. A situation fires it.

| When you see this | Predict this | Move |
|---|---|---|
| You are about to time something | The floor: bytes, arithmetic, launches | 1, 2 |
| Something grows | What happens when it doubles: x1, x2 or x4 | 4 |
| A change saves something | The saving, **and** the bill | 5 |
| A result looks free, or too good | Where the cost hides | 5 |
| Two results come from different machines | The ratio of each to its floor | 3 |
| You change how data is stored or computed | "The output does not change" | 6 |

---

## Where your software instinct is wrong

If you come from software engineering, your instinct has two parts. The
**shape of your search** transfers completely: layers, narrowing, named
patterns. `docs/MAP.md` gives you the layers of an engine in the words of a web
service.

The **assumptions under your search** do not transfer, because you learned them
on CPUs, networks and databases. A GPU breaks six of them:

| Your instinct says | On a GPU engine | Stages |
|---|---|---|
| Each request costs its own share | All the running requests move in the same step. 32 users cost almost the same as 1, and one long prompt slows everybody. | 04, 05, 11 |
| Computation is the expensive part | At batch 1 the arithmetic units wait. The cost is the bytes that move. | 03 |
| The state of one request is small | The memo table of one long conversation can be larger than the whole model. Memory, not CPU, sets how many users fit. | 02, 06 |
| A bug raises an error | A bug gives fluent, wrong text. | 07, 09, 23 |
| Framework overhead is small | Python and launches can cost as much as the GPU work. | 12 |
| The same input gives the same output | A different batch size changes the rounding, and can change the text. | LORE section 12 |

Part 0, section `0_map`, lets you predict each one with your old instinct, and
then measure it.

---

## How the course teaches the method

You cannot learn "when" from a list. You learn it when you choose, and you are
sometimes wrong. So the course gives less help in each Part:

| Parts | The notebook gives you | You give |
|---|---|---|
| 0 and 1 | The trigger, and the question | The number, before the cell runs |
| 2 to 4 | Only the situation: "you are about to change how X is stored" | The question, and the number |
| 5 to 8, and the incidents | Nothing | The question, the number, and the move |

Each prediction cell has the same form:

```python
# PREDICT (move 4: read the shape). Write your number BEFORE you run the next cell.
my_prediction = None      # when the prefix doubles, the time is multiplied by ...
```

When a prediction is wrong, or a sentence confuses you, write it down and
continue:

```bash
./vc note "predicted x4 when the prefix doubled, measured x2: most of a pass is each token through the weights, and that is linear"
./vc note          # read your notes
```

Read the notes at the end of each Part. A wrong prediction that you
understand is worth more than a right one that you guessed.

---

## For the authors of the course

- Start each exercise with what is given and what is asked.
- Give each variable a name that says what it is, of what, and its unit:
  `step_measured_ms`, not `ms`. A ratio names its top and its bottom:
  `step_measured_over_floor`, not `ratio`. A reader must not decode a name
  before they can read the idea. `dev/names.py` checks the notebooks.
- Put a prediction cell before each measurement. Name the move.
- Give the plain words first, and the GPU term second: "a memo table for each
  request. GPU people call it the **KV cache**."
- Give the simple tool before the formal one: the doubling test before
  `polyfit`.
- Each stage has a `layer:` (where it is on `docs/MAP.md`), a `because:`
  (the bill of the stage before) and a `given:`. `tests/test_docs.py` checks
  that all three exist.
  If you cannot write an honest `because:`, the ladder has a gap there.
