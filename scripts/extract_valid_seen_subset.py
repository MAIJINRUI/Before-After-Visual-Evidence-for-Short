import argparse
import json
import zipfile
from pathlib import Path


def load_valid_seen_tasks(limit: int):
    import urllib.request

    split_url = "https://raw.githubusercontent.com/askforalfred/alfred/master/data/splits/oct21.json"
    split = json.loads(urllib.request.urlopen(split_url, timeout=60).read().decode())
    uniq = []
    seen = set()
    for item in split["valid_seen"]:
        task = item["task"]
        if task not in seen:
            seen.add(task)
            uniq.append(task)
        if len(uniq) >= limit:
            break
    return uniq


def extract_subset(zip_path: Path, out_dir: Path, task_paths):
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()
        task_to_members = {task: [] for task in task_paths}

        # Match members that contain "<task>/" anywhere in the archive path.
        for name in names:
            for task in task_paths:
                token = f"{task}/"
                if token in name:
                    task_to_members[task].append(name)

        found = [t for t, members in task_to_members.items() if members]
        missing = [t for t, members in task_to_members.items() if not members]

        for task in found:
            for member in task_to_members[task]:
                zf.extract(member, path=out_dir)

    return {"found_tasks": found, "missing_tasks": missing}


def main():
    parser = argparse.ArgumentParser(description="Extract first N ALFRED valid_seen folders from ZIP")
    parser.add_argument("--zip", required=True, help="path to images.zip")
    parser.add_argument("--out", required=True, help="output directory")
    parser.add_argument("--limit", type=int, default=200, help="number of unique valid_seen tasks")
    parser.add_argument("--report", default="", help="optional output JSON report")
    args = parser.parse_args()

    zip_path = Path(args.zip).resolve()
    out_dir = Path(args.out).resolve()
    tasks = load_valid_seen_tasks(args.limit)
    result = extract_subset(zip_path, out_dir, tasks)
    result["requested_limit"] = args.limit
    result["found_count"] = len(result["found_tasks"])
    result["missing_count"] = len(result["missing_tasks"])

    if args.report:
        report_path = Path(args.report).resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
