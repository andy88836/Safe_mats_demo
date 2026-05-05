from __future__ import annotations

import argparse
from pathlib import Path

from .agents import SafeMOFAgentPipeline
from .llm import LLMConfigurationError


DEFAULT_GOAL = (
    "Discover environmentally compatible MOFs for removing benzene-series "
    "pollutants from water while balancing adsorption capacity, aquatic safety, "
    "water stability, synthesis feasibility, and membrane processability."
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Safe-MOF-Agent MVP.")
    parser.add_argument("--mode", choices=["manuscript", "dataset"], default="manuscript")
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--goal", default=DEFAULT_GOAL)
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument(
        "--require-llm",
        action="store_true",
        help="Fail if OPENAI_API_KEY and model names are not configured in .env.",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Force deterministic fallback even if .env contains API settings.",
    )
    parser.add_argument(
        "--dataset-limit",
        type=int,
        default=None,
        help="Optional number of common sine-matrix candidates to evaluate in dataset mode.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        pipeline = SafeMOFAgentPipeline(
            repo_root=args.repo_root,
            require_llm=args.require_llm,
            disable_llm=args.no_llm,
        )
    except LLMConfigurationError as exc:
        print(f"LLM configuration error: {exc}")
        return 2
    output = pipeline.run(
        mode=args.mode,
        research_goal=args.goal,
        top_n=args.top_n,
        dataset_limit=args.dataset_limit,
    )
    report_dir = pipeline.paths.report_dir
    print(f"Safe-MOF-Agent mode: {args.mode}")
    print(f"LLM enabled: {output['run_metadata']['llm_enabled']}")
    print(f"Candidates evaluated: {len(output['candidates'])}")
    print(f"Report: {report_dir / f'{args.mode}_safe_mof_agent_report.json'}")
    print(f"Decisions: {report_dir / f'{args.mode}_safe_mof_agent_decisions.csv'}")
    if output["decisions"]:
        top = output["decisions"][0]
        print(f"Top candidate: {top['candidate_id']} ({top['decision_class']}, score={top['final_score']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
