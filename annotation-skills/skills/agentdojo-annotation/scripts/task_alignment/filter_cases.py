import json
import shutil
import argparse
from pathlib import Path


def get_num_components(data):
    """
    优先读取 num_components。
    如果不存在，则根据 trajectory 的实际长度计算。
    """
    num_components = data.get("num_components")

    if isinstance(num_components, int):
        return num_components

    trajectory = data.get("trajectory")
    if isinstance(trajectory, list):
        return len(trajectory)

    raise ValueError("Missing valid num_components and trajectory")


def filter_file(path, src_root, out_root, min_components):
    """
    如果 trajectory 长度大于等于 min_components，
    则保持原有相对目录结构复制到输出目录。
    """
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    num_components = get_num_components(data)

    if num_components < min_components:
        return False, num_components

    rel_path = path.relative_to(src_root)
    out_path = out_root / rel_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    shutil.copy2(path, out_path)

    return True, num_components


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Filter normalized AgentDojo trajectories by minimum "
            "number of components."
        )
    )

    parser.add_argument(
        "--src_root",
        type=str,
        required=True,
        help="Input directory containing normalized trajectory JSON files."
    )

    parser.add_argument(
        "--out_root",
        type=str,
        required=True,
        help="Output directory for retained trajectory JSON files."
    )

    parser.add_argument(
        "--min_components",
        type=int,
        default=4,
        help="Minimum number of components to retain. Default: 4."
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete the existing output directory before filtering."
    )

    args = parser.parse_args()

    src_root = Path(args.src_root).resolve()
    out_root = Path(args.out_root).resolve()
    min_components = args.min_components

    if not src_root.exists():
        raise FileNotFoundError(f"Source directory not found: {src_root}")

    if not src_root.is_dir():
        raise NotADirectoryError(f"Source path is not a directory: {src_root}")

    if min_components < 0:
        raise ValueError("--min_components must be non-negative")

    if src_root == out_root:
        raise ValueError("--src_root and --out_root must be different")

    if args.overwrite and out_root.exists():
        print(f"[REMOVE OLD OUTPUT] {out_root}")
        shutil.rmtree(out_root)

    out_root.mkdir(parents=True, exist_ok=True)

    total = 0
    kept = 0
    removed = 0
    errors = 0
    case_results = []

    kept_lengths = []
    removed_lengths = []

    json_files = sorted(src_root.rglob("*.json"))

    for path in json_files:
        total += 1

        try:
            is_kept, num_components = filter_file(
                path=path,
                src_root=src_root,
                out_root=out_root,
                min_components=min_components,
            )

            case_results.append({"source_path": str(path), "status": "kept" if is_kept else "excluded", "reason": "component_count", "num_components": num_components})
            if is_kept:
                kept += 1
                kept_lengths.append(num_components)
                print(
                    f"[KEEP] components={num_components:<3} "
                    f"{path.relative_to(src_root)}"
                )
            else:
                removed += 1
                removed_lengths.append(num_components)
                print(
                    f"[DROP] components={num_components:<3} "
                    f"{path.relative_to(src_root)}"
                )

        except Exception as e:
            case_results.append({"source_path": str(path), "status": "failed", "reason": str(e)})
            errors += 1
            print(f"[ERROR] {path}: {e}")

    (out_root.parent / (out_root.name + "_outcomes.jsonl")).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in case_results))

    print("\n" + "=" * 72)
    print("Filtering finished")
    print("=" * 72)
    print(f"Source directory     : {src_root}")
    print(f"Output directory     : {out_root}")
    print(f"Minimum components   : {min_components}")
    print(f"JSON files scanned   : {total}")
    print(f"Kept trajectories    : {kept}")
    print(f"Removed trajectories : {removed}")
    print(f"Errors               : {errors}")

    if kept_lengths:
        print(f"Kept length range    : {min(kept_lengths)}–{max(kept_lengths)}")

    if removed_lengths:
        print(
            f"Removed length range : "
            f"{min(removed_lengths)}–{max(removed_lengths)}"
        )


if __name__ == "__main__":
    main()
