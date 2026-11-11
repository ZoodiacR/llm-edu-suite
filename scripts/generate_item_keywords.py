"""
Generate LLM keyword profiles for every MovieLens-1M movie (used by notebook 23 for content features and cold start).

The result is committed as data/recsys/ml1m_llm_keywords.json, so the notebook does not need to run this. It takes
about an hour on a 6 GB laptop GPU with qwen2.5:7b, and it can be stopped and resumed: movies already in the file
are skipped.

    python scripts/generate_item_keywords.py                       # qwen2.5:7b, all movies
    python scripts/generate_item_keywords.py --model qwen2.5:3b    # faster, more invented details

The prompt asks for keywords (subgenre, tone, themes, setting, era, audience) and explicitly no names of people:
small models invent directors and actors with confidence, and a wrong name in a profile is worse than none. The
keywords can still be wrong for obscure films; the notebook measures whether they help, it does not assume it.
"""
import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

PROMPT = ("Movie: {title}. Genres: {genres}.\nGive 8 to 12 comma-separated keywords that describe this film for a recommender: "
          "subgenre, tone, themes, setting, era, audience. No names of people. Keywords only.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="qwen2.5:7b")
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "recsys", "ml1m_llm_keywords.json"))
    ap.add_argument("--batch", type=int, default=40, help="movies per batch (the file is saved after each one)")
    ap.add_argument("--concurrency", type=int, default=4)
    args = ap.parse_args()

    from llm_edu_suite.config import get_llm
    from llm_edu_suite.recsys import download_movielens_1m, load_movielens_1m
    ml = load_movielens_1m(download_movielens_1m())
    done = json.load(open(args.out, encoding="utf-8")) if os.path.exists(args.out) else {}
    todo = [(t, g) for t, g in zip(ml.titles[1:], ml.genres[1:]) if t not in done]
    print(f"{len(done)} movies already profiled, {len(todo)} to go, model {args.model}")
    # a short context window keeps the 7B model entirely on a 6 GB GPU; the prompts are ~60 tokens
    llm = get_llm(model=args.model, num_predict=48, num_ctx=512)
    t0 = time.time()
    for b in range(0, len(todo), args.batch):
        chunk = todo[b:b + args.batch]
        outs = llm.batch([PROMPT.format(title=t, genres=", ".join(g)) for t, g in chunk], config={"max_concurrency": args.concurrency})
        for (t, _), o in zip(chunk, outs):
            done[t] = " ".join(o.content.split())
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w", encoding="utf-8", newline="\n") as f:
            json.dump(dict(sorted(done.items())), f, ensure_ascii=False, indent=0)
        print(f"{len(done)}/{ml.n_items}  {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
