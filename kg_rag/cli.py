"""Command line entry point.

    python -m kg_rag.cli setup                 # apply schema, no ingestion
    python -m kg_rag.cli ingest --sample       # ingest the synthetic corpus, no API key needed
    python -m kg_rag.cli ingest --edgar        # ingest real filings (needs SEC_USER_AGENT)
    python -m kg_rag.cli budget --sample       # cost the run without calling a model
    python -m kg_rag.cli ask "who supplies NVIDIA?"
    python -m kg_rag.cli bench --sample        # run the benchmark and write a report
    python -m kg_rag.cli demo                  # ingest, ask the demo questions, print results
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from kg_rag import settings
from kg_rag.ingest import BudgetExceeded, load_sample_corpus, plan_budget


def _corpus(args) -> tuple[list, str]:
    if getattr(args, "edgar", False):
        from kg_rag.ingest import load_corpus_from_edgar
        from kg_rag.sec_client import SecClient

        client = SecClient.from_env()
        records = load_corpus_from_edgar(client, _tickers())
        return records, "edgar"
    return load_sample_corpus(), "sample"


def _tickers() -> tuple[str, ...]:
    from kg_rag.corpus import REGISTRANTS

    return REGISTRANTS


def cmd_setup(args) -> int:
    from kg_rag.index import HashingEmbedder, VectorIndex
    from kg_rag.store import GraphStore

    store = GraphStore.connect(settings.neo4j_uri(), settings.neo4j_auth())
    counts = store.counts()
    print(f"neo4j   {settings.neo4j_uri()}  nodes={counts['nodes']} edges={counts['edges']}")
    store.close()

    index = VectorIndex(
        dsn=settings.postgres_dsn(),
        embedder=HashingEmbedder(dimension=settings.embedding_dimension()),
    )
    index.ensure_schema()
    print(f"pgvector {settings.postgres_dsn()}  chunks={index.count()}")
    index.close()
    return 0


def cmd_budget(args) -> int:
    records, kind = _corpus(args)
    try:
        report = plan_budget(records, settings.max_ingestion_usd())
    except BudgetExceeded as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({
        "corpus": kind,
        "documents": report.documents,
        "chunks": report.chunks,
        "tokens_in": report.tokens_in,
        "tokens_out": report.tokens_out,
        "estimated_usd": round(report.estimated_usd, 4),
        "ceiling_usd": settings.max_ingestion_usd(),
        "extraction": "real" if settings.use_real_models() else "stub (no model calls)",
    }, indent=2))
    return 0


def cmd_ingest(args) -> int:
    from kg_rag.pipeline import build_pipeline, build_sample_gazetteer

    records, kind = _corpus(args)
    ceiling = settings.max_ingestion_usd()
    try:
        plan = plan_budget(records, ceiling)
        print(
            f"planned: {plan.documents} docs, {plan.chunks} chunks, "
            f"~${plan.estimated_usd:.4f}"
        )
    except BudgetExceeded as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2

    gazetteer = build_sample_gazetteer() if kind == "sample" else None
    pipeline = build_pipeline(gazetteer)
    try:
        report = pipeline.ingest(records, cache=not args.no_cache)
        print(json.dumps(report, indent=2))
    finally:
        pipeline.store.close()
        pipeline.index.close()
    return 0


def cmd_ask(args) -> int:
    from kg_rag.pipeline import build_pipeline

    pipeline = build_pipeline()
    try:
        result = pipeline.ask(args.question, as_of=args.as_of)
    finally:
        pipeline.store.close()
        pipeline.index.close()

    if args.json:
        print(json.dumps(result.as_dict(), indent=2))
        return 0

    print(f"Q       {result.question}")
    print(f"route   {result.route}"
          + (f"  template={result.query_type}" if result.query_type else ""))
    if result.cypher:
        print(f"cypher  {result.cypher}")
    if result.refused:
        print(f"REFUSED {result.reason}")
    else:
        print(f"answer  {result.answer}")
    print(f"cites   {', '.join(result.citations) or '(none)'}")
    print(f"took    {result.latency_ms:.1f} ms   generator={result.generator}")
    return 0


def cmd_bench(args) -> int:
    from kg_rag.eval_runner import run_benchmark, write_report

    pipeline = _pipeline_for_bench(args)
    try:
        report = run_benchmark(pipeline, split=args.split, limit=args.limit)
    finally:
        pipeline.store.close()
        pipeline.index.close()

    if args.out:
        write_report(Path(args.out), report)
        print(f"wrote {args.out}")
    print(json.dumps(report["summary"], indent=2))
    return 0


def cmd_demo(args) -> int:
    from kg_rag.pipeline import build_pipeline, build_sample_gazetteer

    records = load_sample_corpus()
    gazetteer = build_sample_gazetteer()
    pipeline = build_pipeline(gazetteer)
    try:
        report = pipeline.ingest(records, cache=False)
        print("=" * 72)
        print(f"INGEST  {report['documents']} docs  {report['chunks']} chunks  "
              f"{report['nodes_written']} nodes  {report['edges_written']} edges")
        print(f"GRAPH   {report['graph']}")
        print(f"EDGES   {report['relationships_by_type']}")
        print("=" * 72)

        questions = [
            "Who are NVIDIA's suppliers?",
            "Which companies compete with NVIDIA?",
            "Who does Intel compete with?",
            "How many relationships does Taiwan Semiconductor Manufacturing Company have?",
            "What is a lithography system?",
            "What is the weather in Paris?",
        ]
        for question in questions:
            result = pipeline.ask(question)
            print(f"\nQ  {question}")
            print(f"   route={result.route} template={result.query_type} "
                  f"{result.latency_ms:.0f}ms")
            if result.refused:
                print(f"   REFUSED: {result.reason}")
            else:
                print(f"   {result.answer[:300]}")
                print(f"   cites: {', '.join(result.citations[:4])}")

        print("\n" + "=" * 72)
        print("STATS", json.dumps(pipeline.stats.as_dict(), indent=2))
    finally:
        pipeline.store.close()
        pipeline.index.close()
    return 0


def _pipeline_for_bench(args):
    from kg_rag.pipeline import build_pipeline, build_sample_gazetteer

    if not args.no_ingest:
        gazetteer = build_sample_gazetteer()
        pipeline = build_pipeline(gazetteer)
        pipeline.ingest(load_sample_corpus(), cache=False)
        return pipeline
    return build_pipeline()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="kg_rag", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("setup", help="apply schema and report store counts").set_defaults(
        func=cmd_setup
    )

    p_budget = sub.add_parser("budget", help="estimate extraction cost, no model calls")
    p_budget.add_argument("--sample", action="store_true")
    p_budget.add_argument("--edgar", action="store_true")
    p_budget.set_defaults(func=cmd_budget)

    p_ingest = sub.add_parser("ingest", help="extract, resolve and write the graph")
    p_ingest.add_argument("--sample", action="store_true", help="use the synthetic corpus")
    p_ingest.add_argument("--edgar", action="store_true", help="fetch real filings from EDGAR")
    p_ingest.add_argument("--no-cache", action="store_true", help="ignore the extraction cache")
    p_ingest.set_defaults(func=cmd_ingest)

    p_ask = sub.add_parser("ask", help="ask one question")
    p_ask.add_argument("question")
    p_ask.add_argument("--as-of", dest="as_of", default=None)
    p_ask.add_argument("--json", action="store_true")
    p_ask.set_defaults(func=cmd_ask)

    p_bench = sub.add_parser("bench", help="run the benchmark")
    p_bench.add_argument("--sample", action="store_true")
    p_bench.add_argument("--split", default="all")
    p_bench.add_argument("--limit", type=int, default=0)
    p_bench.add_argument("--no-ingest", action="store_true")
    p_bench.add_argument("--out", default=None)
    p_bench.set_defaults(func=cmd_bench)

    sub.add_parser("demo", help="ingest the sample corpus and answer demo questions").set_defaults(
        func=cmd_demo
    )

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
