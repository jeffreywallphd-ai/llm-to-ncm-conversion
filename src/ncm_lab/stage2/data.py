"""Publish the bounded Tülu short-answer cohort with a separate locked test artifact."""

from itertools import islice
import json
from ncm_lab.data.records import Example
from ncm_lab.data.selection import fair_rows
from ncm_lab.stage1.data import LOCKED_SCHEMA, publish_qualified


def publish_tulu(run, database, policy, scan, identity):
    selected, eligible = {}, {}
    for split in ("train", "dev", "test"):
        n, groups, source_tokens, answer_positions = database.execute(
            """SELECT count(*),count(DISTINCT node),coalesce(sum(tokens),0),
               coalesce(sum(json_array_length(json_extract(payload,'$.answer_ids'))),0)
               FROM rows WHERE split=?""",
            (split,),
        ).fetchone()
        eligible[split] = {
            "examples": n,
            "groups": groups,
            "prompt_tokens": source_tokens - answer_positions,
            "answer_positions": answer_positions,
            "length_bands": dict(
                database.execute("SELECT band,count(*) FROM rows WHERE split=? GROUP BY band", (split,))
            ),
            "components": dict(
                database.execute("SELECT source,count(*) FROM rows WHERE split=? GROUP BY source", (split,))
            ),
        }
        items = selected[split] = []
        for key in islice(fair_rows(database, split, (1 / 7,) * 7), getattr(policy, f"{split}_examples")):
            run.meter.check()
            payload, group = database.execute("SELECT payload,node FROM rows WHERE key=?", (key,)).fetchone()
            items.append(
                Example(
                    **{**json.loads(payload), "group_id": group, "split": split},
                    schema=LOCKED_SCHEMA if split == "test" else "native_answer_example_v1",
                )
            )
    return publish_qualified(
        run,
        policy,
        selected,
        {
            "canvas": identity["canvas"],
            "source_sha256": identity["source_sha256"],
            "source": identity["dataset"],
            "revision": identity["source_revision"],
            "scan": scan,
            "natural_eligible_pool": eligible,
            "test_exclusion": "exact tokenized prompts from the pinned approved OASST overlap inventory",
        },
        identity,
    )
