# Final-v14 Independent Annotation App

The combined interface exposes all three roles from one local server. Each
author endpoint returns only that author's labels. The adjudicator remains
locked until both author CSVs are complete, then receives only disagreements.

Start the combined interface:

```bash
python3 server.py --role all --port 8765
```

The coordinator can switch roles at `http://127.0.0.1:8765`. For independent
annotation, give each author a role-locked view on that same server:

```text
http://127.0.0.1:8765/?role=author_1&lock_role=1
http://127.0.0.1:8765/?role=author_2&lock_role=1
http://127.0.0.1:8765/?role=adjudicator&lock_role=1
```

Role-specific processes remain available through `--role author_1`,
`--role author_2`, and `--role adjudicator` when annotators must be given
separate URLs.

By default, the server uses
`adjudication_packages/exp_indep_final_static_blind_01`, whose 60 candidate
repairs are independently labeled by both authors. The three automatic
generation failures are coordinator-only records and are not shown to either
author. Use `--package PATH` to work on a copied package. CSV writes use a
temporary file in the same directory followed by an atomic replacement.

Run the backend and API tests with:

```bash
python3 -m unittest -v test_server.py
```
