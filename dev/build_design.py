"""Build the Part 8 notebook "design the engine", before stage 22.

    python dev/build_design.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_practicums import build  # noqa: E402

SETUP = '''
# find the repo root. The directory you start from does not matter.
import ast, sys
from pathlib import Path
ROOT = next(folder for folder in [Path.cwd(), *Path.cwd().parents]
            if (folder/'cudalib').is_dir())
sys.path.insert(0, str(ROOT))
'''

JOBS = [
    ('admit', 'decide which waiting request starts, and when'),
    ('budget', 'decide how many tokens each sequence gets in one step'),
    ('blocks', 'give a sequence the blocks for its new tokens, and take them back'),
    ('prefix_lookup', 'find the cached blocks of a new prompt'),
    ('prefix_insert', 'put the full blocks of a sequence into the prefix cache'),
    ('victim', 'choose the sequence to preempt, and put it back in the queue'),
    ('flat_batch', 'turn the plan of a step into one flat batch for the model'),
    ('forward', 'run the model on the flat batch'),
    ('sample', 'turn the logits into tokens, with the parameters of each request'),
    ('detokenize', 'turn the new tokens into text for the stream'),
    ('stop', 'decide that a sequence is finished: stop token, stop string, max_tokens'),
    ('abort', 'end a request that the client left, at any moment'),
    ('metrics', 'record TTFT, the time between tokens, and the preemptions'),
    ('speculate', 'propose draft tokens, and count the ones that the model accepted (stage 25)'),
    ('guide', 'mask the logits that break the JSON schema (stage 26)'),
]

EX2_HELPER = '''
JOBS = {
''' + '\n'.join(f'    {name!r}: {text!r},' for name, text in JOBS) + '''
}

# Name a component for each job. Use your own names: a component is a class,
# or a function, that you would write. Two jobs can have the same owner.
OWNER = {
''' + '\n'.join(f'    {name!r}: None,' for name, _ in JOBS) + '''
}
'''

EX2_SOLUTION = '''
JOBS = {
''' + '\n'.join(f'    {name!r}: {text!r},' for name, text in JOBS) + '''
}

# One possible answer: the split of stage 22.
OWNER = {
    'admit': 'Scheduler',
    'budget': 'Scheduler',
    'blocks': 'KVBlockManager',
    'prefix_lookup': 'KVBlockManager',
    'prefix_insert': 'KVBlockManager',
    'victim': 'Scheduler',
    'flat_batch': 'ModelRunner',          # stage 21
    'forward': 'ModelRunner',
    'sample': 'LLMEngine',                # with the sampler of stage 13
    'detokenize': 'Sequence',             # with the detokenizer of stage 14
    'stop': 'LLMEngine',
    'abort': 'LLMEngine',
    'metrics': 'LLMEngine',               # with the metrics of stage 16
    'speculate': 'LLMEngine',             # the hooks _propose and _process
    'guide': 'LLMEngine',                 # the hook _mask
}
'''

CHECK_OWNERS = '''
### Run this cell. It checks only that your design is complete.

missing = [job for job, owner in OWNER.items() if not owner]
assert not missing, f'these jobs have no owner yet: {missing}'
by_owner = {}
for job, owner in OWNER.items():
    by_owner.setdefault(owner, []).append(job)
for owner, jobs in sorted(by_owner.items(), key=lambda item: -len(item[1])):
    print(f'{owner:45s} {", ".join(jobs)}')
print(f'\\n{len(by_owner)} components for {len(OWNER)} jobs')
'''

EX3_HELPER = '''
# Write one step of your engine as an ordered list of calls, in words or in
# code. Name the owner of each call. For example:
#   ('Scheduler', 'plan the step: tokens for each sequence, within the budget')
STEP = [
]
'''

EX3_SOLUTION = '''
STEP = [
    ('LLMEngine', 'drop the requests that the client aborted, and free their blocks'),
    ('Scheduler', 'plan: decodes first, then prefill chunks, then admissions, within the budget'),
    ('Scheduler + KVBlockManager', 'grow the blocks of each planned sequence; preempt the newest if they run out'),
    ('KVBlockManager', 'for a new admission, take the cached prefix, so its computed tokens start there'),
    ('ModelRunner', 'build one flat batch from the plan, and run the model'),
    ('LLMEngine', 'count the computed tokens of each sequence (_process)'),
    ('LLMEngine', 'for each sequence that finished its pending tokens: mask (_mask), then sample'),
    ('Sequence', 'append the token, and detokenize it'),
    ('LLMEngine', 'check the stop conditions; finish, free the blocks, cache the full blocks'),
    ('LLMEngine', 'record the metrics, and return (rid, new_text, finished) for the stream'),
]
'''

SHOW_GIVEN = '''
### Run this cell after you wrote your design. It reads the split of stage 22.

source = (ROOT / 'app' / 's22_engine.py').read_text()
tree = ast.parse(source)
for node in tree.body:
    if isinstance(node, ast.ClassDef):
        methods = [item.name for item in node.body if isinstance(item, ast.FunctionDef)]
        print(f'{node.name}: {", ".join(methods)}')
'''

CELLS = [
    ('md', '''
# Design the engine, before you read the given one

You have every part: the block allocator and prefix cache (stages 06 and 09),
the scheduler rule (stages 10, 11 and the notebook before this one), the model
runner on the paged cache (stage 21), the sampler (13), the detokenizer (14),
the metrics (16), speculation (17) and the JSON mask (19).

Stage 22 joins them into one engine. Its file gives you a split: four classes
and about thirty methods with fixed names. If you open it first, you fill in
someone else's design, and you never find out why it has that shape. So design
your own first. Then compare.

There is no single correct design. A design is good when you can walk every
hard event through it and say which component changes which state, in which
order. The exercises below are those walks.

This notebook needs no GPU. Do not open `app/s22_engine.py` until Exercise 5.
'''),
    ('md', '''
# Exercise 1: the state of one request

Write down every field that one request needs, from its arrival to its last
token. For each field, say who writes it. Hint: the scheduler of the notebook
before this one needed only a few numbers. The engine needs more: think about
the text, the sampling, the blocks, the prefix cache and the metrics.

**Your fields:**
'''),
    ('md', '''
# Exercise 2: an owner for each job

Each job below needs exactly one owner. Two jobs can share an owner. A job
with two owners is a bug that waits for its moment: both will write the same
state.
'''),
    ('code', EX2_HELPER, EX2_SOLUTION),
    ('code', CHECK_OWNERS),
    ('md', '''
# Exercise 3: one step, in order

Write the calls of one `step()` in order. The order is the design: most bugs of
an engine are a correct call at the wrong moment.
'''),
    ('code', EX3_HELPER, EX3_SOLUTION),
    ('md', '''
# Exercise 4: walk five hard events through your design

For each event, write which components change which state, in which order.
Then write what breaks if two of those calls swap.

1. **A cached prefix.** A request arrives, and the first 30 blocks of its prompt
   are in the prefix cache. What is its `computed` count at admission? Why must
   at least one prompt token stay pending?
2. **The pool runs out.** A running sequence needs one more block, and none is
   free. Two sequences share blocks with the one that your design preempts.
   What happens to the shared blocks?
3. **An abort in the middle of a chunked prefill.** The client leaves while the
   second of four prefill chunks is in the current step. Where do you free the
   blocks: before the step, during it, or after it?
4. **A stop string inside a speculative block.** Speculation accepts three
   tokens in one step, and the stop string completes at the second one. What
   does the client see, and what happens to the third token and its KV?
5. **A new feature without a copy.** Stage 26 must [mask the logits](../../GLOSSARY.md#logit-mask) for JSON
   mode. Where does it attach to your design, without a copy of `step()`?

**Your walks:**
'''),
    ('md', '''
# Exercise 5: compare with the given split

Now open `app/s22_engine.py`, and read its docstring. The cell below lists its
classes and methods.
'''),
    ('code', SHOW_GIVEN),
    ('md', '''
Answer these:

1. Where does your split differ from the given one? For each difference,
   which of the five walks of Exercise 4 does your design handle better, and
   which worse?
2. The given design puts the prefix cache inside `KVBlockManager`, not in its
   own class. What does that make easy, and what does it make hard?
3. The given `LLMEngine` has five hooks (`_propose`, `_chunks_for`,
   `_process`, `_mask`, `_on_token`) for stages 25 and 26. Did your design
   need hooks? If not, how did your Exercise 4.5 attach the JSON mask?
4. The given scheduler admits nothing in a step that preempted. Find the walk
   of Exercise 4 where that rule matters.

Keep your design next to the given one while you build stage 22. When a check
fails, ask first which component owns the state that is wrong.

    ./vc guide 22
'''),
    ('solution', '''
### One answer, and the reasons for it

The solution cells above fill in the split of stage 22. Its reasons:

- **The state of a request** is in `Sequence`: the prompt ids and the output
  ids, `num_computed` (the tokens with K and V in the cache), the block table,
  the sampling parameters and the seed, the detokenizer, the stop strings, and
  the times for the metrics. The scheduler writes `num_computed` and the
  blocks; the engine writes the output, the text and the times.
- **The prefix cache lives in `KVBlockManager`**, because a cached block and an
  allocated block are the same physical block with a reference count. One
  owner of the reference counts means that no second component can free a
  block that the first still uses (Part 3, the mystery pool A).
- **The scheduler knows nothing about tokens or text.** It sees pending token
  counts and blocks. So it runs in a unit test with no model, as in the
  challenge after this one.
- **Speculation and the JSON mask attach through hooks**, so that stages 25
  and 26 are subclasses and never copy `step()`. A copy of `step()` is two
  engines that drift apart.
- **The walks.** (1) A cache hit sets `num_computed` to the cached tokens, and
  leaves the last prompt token pending, because only a forward pass on it makes
  the logits of the first output token. (2) A preemption releases the victim's
  references; a shared block stays alive while another sequence holds it. (3)
  An abort frees the blocks between steps, never during one: the step in flight
  still writes into them. (4) The client sees the text up to the stop string;
  the third token is thrown away, and its KV is in blocks that the finish
  releases. (5) The mask is a hook between the logits and the sampler.
- **No admission in a step that preempted.** Without it, the scheduler can
  preempt a sequence and admit a new one in the same step, and the new one then
  takes the blocks that the preemption freed for a running sequence: the loop
  of Part 4, Ticket 5.
'''),
]

if __name__ == '__main__':
    build([('code', SETUP)] + CELLS,
          'course/Part8_TheCapstone/1_engine/part8_eng_3_CCdesignTheEngine_helper.ipynb',
          'course/Part8_TheCapstone/1_engine/solutions/part8_eng_3_CCdesignTheEngine.ipynb',
          'One engine from the parts', 'design the engine, then compare')
    print('the design notebook is built')
