# Blast Radius: change-aware test selection for sqlglot

Entry for the RWX Blast Radius track (Builders Buildathon, Nov 2026). For each
patch against sqlglot `eb11aec`, it prints which tests to run and how to split
them across workers.

## How it decides

1. **AST diff, not text diff.** `astdiff.py` walks the old and new syntax trees
   together and reports the smallest subtrees that differ. Comments and
   formatting change nothing, so they select nothing. Other edits proved to
   change nothing are dropped too:
   - *Renamed locals.* A function whose only difference is a consistent,
     one-to-one renaming of its local variables counts as unchanged only if
     scope analysis shows every renamed name is a non-parameter local of
     that function, nothing in it can see its frame (`locals`, `eval`,
     `vars`, ...), and both versions compile to identical bytecode apart
     from the names.
   - *Reordered imports.* A contiguous run of plain stdlib or third-party
     imports, each binding a different name, may be permuted. Imports of
     sqlglot or the tests, relative, star and `__future__` imports, never
     count as reordered.
   - *Moved definitions.* Undecorated functions with constant defaults and no
     annotations may swap places within a run of such definitions between
     identical statements, in a module or class body.
   - *Overwritten defs.* In a module or class body, a def that the very next
     statement rebinds with another def of the same name (an overload stub
     before its implementation) is never seen: no metaclass in sqlglot
     defines `__prepare__`, and the next def's header must not read the
     name. Building it must do nothing else: no decorator other than
     `typing.overload` (which only records the stub in a registry nothing
     reads), literal defaults only, and no annotation left to evaluate.
     Such defs are left out of the comparison. When they differ, the
     snapshot child still imports the patched module, so a new import error
     is caught.
   - *Renamed module-level names.* A function or constant bound once at
     module level and renamed together with every use in its file is
     unchanged when neither name appears in any other file a test run can
     import or read, nothing in the file reads names by string or through
     `globals()`/`vars()`, and (for a function) every use is a direct call,
     so nothing can see its `__name__`. Neither this nor the renamed-locals
     rule applies in a folder whose listing test reads code objects
     (`test_typing` compares the `sqlglot/typing` annotators by `co_names`
     and `co_varnames`).
   - *Renamed private methods.* A method renamed together with every use in
     its file is unchanged when both names are plain private names (not
     dunder or `_sunder_` hooks, not `*_sql`, which the generator dispatches
     on by name), every use is its def or an attribute load on `self`, `cls`
     or `super()`, the classes that define it have only sqlglot bases, and
     neither name appears in any other file.
   - *New class members nothing else names.* A member the patch adds to a
     class (a constant or a method) under a name that appears in no pristine
     file can be reached only through the patch's own new text (say, a call
     from a changed method), and those changes are charged on their own. The
     member itself charges nothing when the class's whole MRO is sqlglot code
     (no Enum, dataclass, NamedTuple or Protocol) and the name is not dunder,
     `_sunder_` or `*_sql`. A new public `X_sql`
     method on a generator is reached only through the generator's dispatch
     (keyed by the expression class `X`) or a `node.key + "_sql"` lookup.
     That holds when no ancestor defines the name, every subclass resolves it
     to the new method or its own override, and every mention of the name in
     sqlglot and the tests is a `def` header (other generators' own
     methods). Before the patch, a node of class `X` (a function, which fell
     back to `function_fallback_sql`) had its class's `arg_types` read. So
     such a method charges only the tests that read something through class
     `X` and used that generator family.
   - *Class annotations.* Adding, removing or retyping an annotated class
     attribute (`FLAG: bool = False`) also changes the class's
     `__annotations__`. Nothing in sqlglot or its tests reads a class's
     annotations at run time (no `get_type_hints`, `inspect.get_annotations`
     or dataclass on such classes), and what class creation derives from them
     is fingerprinted on its own, so on a class whose whole MRO is sqlglot
     code (no Enum, dataclass, NamedTuple, TypedDict or Protocol) that entry
     charges nothing (nor does it keep the TPC tests from being judged by
     front-end fingerprints). The attribute's own value is a changed
     attribute like any other.
   - *New modules nothing imports.* Python imports a module only by its name,
     and every dynamic import in sqlglot and its tests builds that name from
     text found somewhere (a dialect string, an attribute name, a fixed
     table). A new module in an existing sqlglot folder whose name appears in
     no other file, old or patched, and that no test sees by listing a folder,
     never runs. Names pytest might collect (`test*`, `*_test`, `conftest`)
     never count.
2. **Runtime changes → line coverage.** A difference inside a function or
   lambda body only matters to tests that execute it. A lambda's body runs
   in its own frame when the lambda is called, so a change inside it is
   charged to the tests that ran the lambda's body lines, not every line of
   the statement around it (one entry of a dict holding dozens of lambdas,
   say). Likewise an `except` clause's type and name are evaluated only
   while an exception is matched against it, which fires that clause's own
   line (not the `try` line), and its body runs only after that: a change
   there is charged to the tests that ran the clause. Code that reads a lambda's code object instead of calling it is
   only `test_typing`, which also reads the source files it checks. `tracer.py` ran every
   test of the pinned commit in its own forked process with `sys.monitoring`
   and recorded the lines each one executed (process-pool work included; see
   `inline.py`). A test run alone can miss one thing a full pytest run does:
   `tests/test_dialect_entry_points.py` empties the dialect registry
   (`Dialect._classes`) in `setUp` and `tearDown`, so any test pytest runs
   after it in the same process looks its dialects up again through
   `_try_load`. A second trace starts every test with the registry emptied
   and keeps the lines tests run only then (`maps/regcov.json.gz`, all in
   `dialects/dialect.py`). A change to one of those lines keeps those tests
   unverified, because a child runs each test from a full registry. (One
   sample of this: narrowing the `except ImportError` in `_try_load` breaks
   a generator test only when it runs after the entry-point tests.)
3. **Import-time changes → measured state diff.** Class bodies, module
   constants, def headers and decorators act at import time, and sqlglot's
   metaclasses derive many structures from them (tries, inverse mappings,
   generator transforms). Instead of guessing, a child imports the patched
   sqlglot and fingerprints every class attribute and module global
   (`fingerprint.py`). It compares them with the pristine fingerprint taken at
   build time. Nested containers are fingerprinted eight levels deep, except
   tries (one character per level), which are fingerprinted to the end of
   every key. A new keyword `"PARTITION BY RANGE"` changes how every text
   starting with `PARTITION BY` is scanned, twelve levels down the
   tokenizer's keyword trie. Each changed `(class, attribute)` maps to the
   tests that read that attribute through the class or a subclass. The tracer records those
   reads by wrapping every class attribute in a recording descriptor. Each
   changed global maps to the tests that execute a line referring to it.
   One registry gets a closer look: a new expression class adds a key to
   `exp.EXPR_CLASSES`, which every test that builds a generator reads (in
   `_build_dispatch`). When the only change to it is new keys for new
   classes, it charges nothing. The new dispatch entries are keyed by the new
   classes, so they matter only for nodes of those classes, and a test can
   only get such a node through another changed lookup (the parser's
   `FUNCTIONS`, `FUNCTION_BY_NAME`, the new names), each charged on its own.
   The same holds when the only change is lost keys of expression classes
   that are gone (a reverted or deleted node type): only nodes of those
   classes used their entries, and a test gets one only through the class,
   its name or a changed lookup, each charged on its own. That needs every
   class left to keep its key, and none of them to hold a gone class's key
   (it would take over the entry). Either way the patch may change
   `generator.py` elsewhere, as long as `_build_dispatch` keeps its pinned
   source and the patched files name `EXPR_CLASSES` nowhere else.
   Tokenizer settings get a closer look too. A `Tokenizer` hands every
   setting to a fresh `TokenizerCore` each time one is built, so the trace
   shows every tokenizing test of a dialect family reading every setting.
   Where a setting's value actually goes is read from the source: when its
   only mentions are that hand-off (`field=self.SETTING`), class bodies,
   class-creation hooks and code that runs only at test time, and the core
   reads its field only as `self.field` after a single plain store in
   `__init__`, the change is charged to the tests that ran the core's reads
   of that field or those other lines, plus the test files that name the
   setting. Both this set and the class-read set cover the change, so the
   cheaper of the two is used.
   New default values get a closer look as well. Python evaluates a def's
   defaults once, at import time, so a changed default changes the function
   object that the fingerprint sees. Take an undecorated def whose header
   differs only in the values of literal defaults (same parameters, kinds,
   annotations and return). Its new value can be seen only by a call that
   leaves that argument out, and such a call runs the function's body.
   Nothing in sqlglot or its tests reads `__defaults__` or a signature's
   defaults: `inspect` is used there for parameter names only. So the
   function, as a class attribute or a module global, is charged to the
   tests that ran its body (coverage) and to those that read the attribute
   through its class family. This does not apply in folders whose listing
   test reads code objects.
4. **Operator swaps → value profile.** When the only difference in a
   comparison is its operator (`is` for `==`, `<` for `<=`, `is not` for `!=`,
   ...), executing the line is not enough to change anything: the new
   operator must also give a different result on the same operands. Until
   that first different result, the patched program runs exactly like the
   original. `wmtrace.py` imports sqlglot with every one-operator comparison
   rewritten to also evaluate the alternative operators. It runs each test in
   its own process and records, per comparison site, which alternatives ever
   gave a different result (or raised differently) in that test. A swap is
   charged only to those tests. Line coverage still applies to tests whose
   instrumented run cannot be trusted, and to sites the profile does not
   cover. The profile is the union of a native run and an amd64 run.
   Swapping one `and` for `or` (or back) in a chain gets the same
   treatment from `bwtrace.py`. It rewrites every `and`/`or` chain inside a
   function into a call that evaluates the operands exactly as Python does,
   then works out what each single swap would have done with the same
   values: `a and b and c` with its second `and` swapped reads
   `(a and b) or c`. The swapped chain may need an operand the original
   skipped. The tracer evaluates it only through a guarded copy that refuses
   anything that could change state: calls other than a few builtins,
   `dict.get`, `str` methods and `Expression.text`, arithmetic on anything but
   builtin numbers, a property whose getter stores something, a cached
   property, a module's `__getattr__`. An operand the original evaluated but
   the swap would skip must pass the same guard. Where only the chain's truth
   is used (an `if` or `while` test, `not`, `bool()`, `assert`), equal truth
   is enough; elsewhere the swapped value must be the very same object.
   Anything it cannot decide counts as a difference, and a swap is charged
   only to the tests in which it ever made one.
5. **Query-level verification for the TPC tests.** `test_execute_tpcds`
   and its TPC-H siblings loop over a fixture's queries. `qtrace.py` records
   which queries execute each line. When a change reaches only some of them,
   the test can be verified on those queries alone. The optimizer tests that
   check a rule against a fixture file get the same treatment per fixture
   pair (`cftrace.py`, `cfsel.py`), unless the change reaches code the test
   runs outside its pairs (a schema it builds for every pair, say): then it
   runs whole.
   When a change reaches only the front end of a TPC executor test (parse,
   optimize, plan, transpile), the selector recomputes what that front end
   hands each query's execution and compares it with the pinned commit's
   (`tpcfp.py`, `ftrace.py`): the same hand-over means the same result. A
   plan may also match without the expression slots (type, metadata,
   comments) that executing that query never reads except to copy them.
   `ftrace_lean.py` records which slots each query reads, under three hash
   seeds. The untouched execution side then takes the same steps.
6. **Data files → file reads.** SQL fixtures and other files map to the tests
   that opened them (or whose module opened them while importing). A new
   top-level module named like a standard-library module (or
   `sitecustomize`) keeps every test: it comes first on `sys.path`.
7. **Verify cheap candidates.** Candidates that cost little to run are run on
   a symlink copy of the patched tree in a forked child, and kept only if they
   fail. Expensive ones are kept unverified when that is cheaper. The cost
   model uses each test's measured CPU and its class setup, and a prior
   chance that a candidate fails (one constant per kind of reason it was
   chosen: 0.2 for code and import-time state, 0.5 for test-side changes,
   0.9 for data files). A child runs with one hash seed and pytest with
   others, so a patch that takes a `sorted(` or `.sort(` out of sqlglot code
   (output may then follow set order), or adds a set (a display, a
   comprehension, `set(` or `frozenset(`), a `hash(` or `id(` call or a use of
   `random` to sqlglot or test code, keeps its candidates unverified. Every
   verified test runs under its own timer (three times its measured cost plus a
   margin; 15 s for class setup); a test that runs out of time is kept,
   because a patch that makes a test hang makes it fail. A test stops at
   its first failing subtest: it has failed, and the rest would only cost
   time. The tests that hand fixture pairs or TPC queries to a process pool
   run in the child with an in-process pool whose tasks run, in the order
   they were submitted, when their result is first asked for; so a test
   that fails at one pair never computes the pairs after it, and a test
   that passes runs every task as before. Pairs marked with a dialect whose
   own file the patch changes are checked first. A patch can also
   make a test allocate without end, faster than any timer: a thread in each
   child stops it once it holds 1.3 GB more than at the fork (address space
   is capped too, far above that), and that commit keeps every candidate, so
   two children never take the container past its memory limit.
   `TestExecutor`'s class setup parses sixteen TPC CSV files in pure Python
   (about 1 s of CPU) before any of its tests can run. Only the test file,
   `tests/helpers.py`, the CSV bytes and the standard library decide the
   rows it builds, so the image build runs it once on the pinned commit and
   stores them (`mkxcache.py`, which checks that the stored path rebuilds
   identical class state). A child uses them only when every one of those
   files in its tree is the pinned one; it still builds each `Table` with
   the patched class, and opens the duckdb views at first use.
   The cost model then prices that class setup at 15% of its measured cost.
   Importing the patched code has a CPU-time limit: 5 s for sqlglot (the
   snapshot; the slowest normal case takes 1.2 s) and 8 s for the test
   modules before verifying. Some modules parse SQL while they import,
   so a patch that makes the parser loop forever hangs that import, and it
   hangs pytest's collection the same way: such a commit gets every test
   within seconds instead of holding a child until the parent kills it.
   Most children never need their own `import sqlglot`. When a patch leaves
   alone every module that import runs (`maps/core.json`) and adds, deletes
   and renames nothing, changing only other existing sqlglot modules or
   test-side files, the parent imports pristine sqlglot once and forks
   those children after the others. Each points the imported modules'
   files and package paths at its patched tree, so every other module, the
   patched ones among them, loads from there, exactly as after a fresh
   import.
8. **Pack.** The final set goes onto 1–8 workers (longest first), choosing
   the worker count that minimises busiest-worker time + 2 s per worker.
   Once every line is printed and every child reaped, the process exits
   without the interpreter's teardown (freeing every module and object costs
   CPU and changes nothing in the output).

When anything is uncertain (unparseable patch, import failure, pytest
configuration change, child error or time budget), it falls back to a safe
superset. One such case is a test-side module (a test file, a package
`__init__`, a helper the tests import, or a new file) whose import-time
statements change in a way that could touch shared state (patching a sqlglot
class, defining a dialect subclass, setting an environment variable) or
change pytest's verdict (an xfail or timeout mark). pytest imports every
test module while collecting, in every worker, so shared state can break
tests in any file. Imports, plain definitions,
test classes, skip and `mock.patch` decorators and call-free constants are
known not to. A test module that changes only inside its test classes, and
there only in plain methods (added, removed or edited, with no decorators,
constant defaults, and annotations that cannot fail when Python 3.13 runs the
`def`; nothing the class body runs names a changed method), charges the tests
of those classes and their subclasses in the module, plus the test files
that may use one of those classes. When every changed method is a test
(`test*`) that nothing else names, in the module or any other test file, only
those tests are charged: removing `test_conditionless_join` from `TestParser`
charges that test, not `test_max_nodes` and the other 79.
Files that import only other names from it are left out: every dialect test
file imports just `Validator` from `tests/dialects/test_dialect.py`, so a
method added to `TestDialect` does not reach them.

The main process guards itself the same way. A patch that keeps the analysis
of one commit busy for 20 s gets every test for that commit. If anything in
the main process raises, or the batch nears its time limit, every commit not
yet printed still gets exactly one line: its selection so far, else every
candidate, else every test. A process a test leaves running cannot stall it
either: when a child exits, whatever it left in its process group is killed,
and its answer is read without waiting on a pipe something else still holds.

No answers are worked out ahead of time: there is no table from patches or
kinds of edit to failing tests. The maps in `maps/` are built only from the
pinned commit (coverage, attribute reads, file reads, imports, name
references).

## Files

| File | Role |
| --- | --- |
| `selector.py` | Runtime entry point (`run.py` runs it from its compiled bytecode) |
| `astdiff.py` | Semantic AST diff |
| `fingerprint.py` | Import-time state fingerprints and diff |
| `patchapply.py` | Unified-diff parser and applier (stdlib only) |
| `inline.py` | In-process ProcessPoolExecutor with pickle round trips, tasks run when their result is asked for |
| `xsetup.py`, `mkxcache.py` | `TestExecutor`'s class setup from rows stored at image build (`xcache.bin`) |
| `tracer.py`, `modlines.py`, `build_maps.py`, `mksnap.py` | Build-time tools that produce `maps/maps.json.gz` and `snap.json` from the pinned commit |
| `wmtrace.py`, `build_wm.py` | Build-time: comparison-swap profile, `maps/wm.json.gz` |
| `bwtrace.py`, `build_bw.py` | Build-time: and/or-swap profile, `maps/bw.json.gz` |
| `build_core.py` | Build-time: the modules `import sqlglot` runs, `maps/core.json` |
| `build_regcov.py` | Build-time: lines tests run only with the dialect registry emptied, `maps/regcov.json.gz` |
| `qtrace.py`, `build_split.py` | Build-time: per-query coverage of the TPC tests, `maps/split.json.gz` |
| `tpcfp.py`, `ftrace.py`, `build_fe.py`, `ftrace_lean.py` | Front-end fingerprints of the TPC executor tests, `maps/fe.json.gz` |
| `cftrace.py`, `build_cf.py`, `cfsel.py` | Per-pair coverage of the fixture-driven optimizer tests, `maps/cf.json.gz`; `cfsel.py` runs such a test on chosen pairs |

## Rebuilding the maps

```sh
# inside the image (python:3.13-slim + requirements.lock + sqlglot at eb11aec on PYTHONPATH)
PYTHONHASHSEED=0 python tracer.py /opt/sqlglot tests.json out/trace 6
TRACE_CALIB=1 PYTHONHASHSEED=0 python tracer.py /opt/sqlglot tests.json out/calib 2
python modlines.py /opt/sqlglot tests.json out/modlines.json
CALIB_DIR=out/calib python build_maps.py /opt/sqlglot tests.json out/trace out/modlines.json maps/maps.json.gz
PYTHONHASHSEED=0 python wmtrace.py /opt/sqlglot tests.json out/wm 8
python build_wm.py tests.json out/wm out/calib maps/wm.json.gz
PYTHONHASHSEED=0 python bwtrace.py /opt/sqlglot tests.json out/bw 8
python build_bw.py tests.json out/bw out/calib maps/bw.json.gz
python build_core.py /opt/sqlglot maps/core.json
TRACE_CLEARREG=1 PYTHONHASHSEED=0 python tracer.py /opt/sqlglot tests.json out/trace_cr 6
python build_regcov.py out/trace out/trace_cr maps/regcov.json.gz
PYTHONHASHSEED=0 python qtrace.py /opt/sqlglot out/qtrace.json 4
python build_split.py out/qtrace.json maps/split.json.gz
PYTHONHASHSEED=0 python cftrace.py /opt/sqlglot out/cftrace 10
python build_cf.py out/cftrace maps/cf.json.gz
PYTHONHASHSEED=0 python ftrace.py /opt/sqlglot out/ftrace.json
python build_fe.py out/ftrace.json maps/fe.json.gz
python ftrace_lean.py /opt/sqlglot maps/fe.json.gz
```

## Third-party code

None beyond the pinned environment in `requirements.lock` (installed from
PyPI) and sqlglot itself (downloaded from its own repository at `eb11aec`).

## License

MIT
