# Resumable Community Report evidence-part summarization

A Community Report can consist of many evidence partitions and a final
synthesis call. A transient OpenCode / OpenAI-compatible model timeout must
not force completed, identical evidence partitions to be summarized again.

The report-generation CLI now writes a checkpoint immediately after every
successfully validated partition summary, before requesting the next one.

## Usage

Run with the existing OPENAI_COMPATIBLE_BASE_URL / MODEL credentials:

    python scripts/generate_cached_community_reports.py \
      --data-repo /path/to/graphrag-drift-data \
      --checkpoint-dir /path/to/persisted-checkpoints

If --checkpoint-dir is omitted, the default is
DATA_REPO/.community_report_checkpoints. This directory must survive runner
failure (for example via a GitHub Actions artifact) to be useful across runs.

The source data-repo and its versioned Community Report manifest are unchanged
until a complete final report has been generated and validated.

## Safety and invalidation

A checkpoint is accepted only when all of the following identities match:

- original dump zip SHA-256
- Community Context SHA-256
- community ID and evidence-part index
- the entire evidence-part JSON
- configured generator name and max-part input size
- partition prompt-version marker

The checkpoint file itself records the SHA-256 of the entire deterministic
identity. Corrupted or mismatched checkpoint files cause an error; they are
never used to answer a legal question.

Successful re-use skips only the partition summary call. The final Community
Report must still be synthesized if it was not completed in the previous run.

After a different source dump, changed law or new context, new checkpoint keys
are generated and stale caches are not reused.

## CI behavior

GitHub Actions must archive the checkpoint directory with if: always() so
that work from an interrupted job is not lost. To resume on another runner,
restore the archive into the same configured checkpoint directory and rerun
the generation command.

The checkpoint does NOT prove the truth of LLM-generated legal summaries,
only that identical requests are not repeated. Temporal source auditing and
direct article coverage validation are separate mandatory gates.
