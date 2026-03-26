"""
Build and extend a layup database by enumerating layups.

Default sampling target
-----------------------
- Candidate angles: [-45, 0, 45, 90]
- Symmetric layup: True
- Strong requirement: True/False
- Ply range: [8, 20]

Database columns
----------------
- layup_id: global unique integer ID
- n_ply: number of plies
- sub_id: unique integer ID within each n_ply group
- stacking: stacking sequence (JSON string of angle list)

Access patterns
---------------
- by layup_id
- by (n_ply, sub_id)
"""

from __future__ import annotations

import itertools
import json
import multiprocessing
import os
import time
from pathlib import Path
from typing import Dict, Iterable, Iterator, List

import numpy as np
import pandas as pd

from lamkit.requirements import EngineeringRequirements
from lamkit.analysis.laminate import Laminate


DATA_PATH = os.path.join(Path(__file__).resolve().parents[3], "data")
DEFAULT_DATASET_PATH = os.path.join(DATA_PATH, "layup_database.csv")


def load_layup_database(dataset_path: str) -> pd.DataFrame:
    """
    Load existing layup database, return empty DataFrame if file does not exist.
    
    The database file is a CSV file with the following columns:
    - layup_id: global unique integer ID
    - n_ply: number of plies
    - sub_id: unique integer ID within each n_ply group
    - stacking: stacking sequence (JSON string of angle list)
    """
    if dataset_path is None:
        return pd.DataFrame(columns=["layup_id", "n_ply", "sub_id", "stacking"])
    
    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        return pd.DataFrame(columns=["layup_id", "n_ply", "sub_id", "stacking"])

    df = pd.read_csv(dataset_path)
    required = {"layup_id", "n_ply", "sub_id", "stacking"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(
            f"Invalid database format in {dataset_path}. Missing columns: {sorted(missing)}"
        )

    return df[["layup_id", "n_ply", "sub_id", "stacking"]].copy()


def get_layup_by_id(layup_id: int,
    dataset_path: str | Path = DEFAULT_DATASET_PATH) -> List[float]:
    """Query stacking sequence by layup_id."""
    df = load_layup_database(dataset_path)
    matched = df.loc[df["layup_id"] == int(layup_id)]
    if matched.empty:
        raise KeyError(f"layup_id={layup_id} not found.")
    raw = matched.iloc[0]["stacking"]
    return json.loads(raw) if isinstance(raw, str) else raw


def get_layup_by_n_ply_and_sub_id(n_ply: int, sub_id: int,
    dataset_path: str | Path = DEFAULT_DATASET_PATH) -> List[float]:
    """Query stacking sequence by (n_ply, sub_id)."""
    df = load_layup_database(dataset_path)
    matched = df.loc[(df["n_ply"] == int(n_ply)) & (df["sub_id"] == int(sub_id))]
    if matched.empty:
        raise KeyError(f"(n_ply, sub_id)=({n_ply}, {sub_id}) not found.")
    raw = matched.iloc[0]["stacking"]
    return json.loads(raw) if isinstance(raw, str) else raw


def iterate_symmetric_layups(
    candidate_angles: List[float], n_ply: int
    ) -> Iterator[List[float]]:
    """Yield symmetric layups one at a time (memory-safe for large n_ply)."""
    if n_ply % 2 != 0:
        raise ValueError("n_ply must be even.")
    for half in itertools.product(candidate_angles, repeat=n_ply // 2):
        h = list(half)
        yield h + h[::-1]


def symmetric_layup_candidate_count(candidate_angles: List[float], n_ply: int) -> int:
    """Number of symmetric candidates from brute-force product over half thickness."""
    if n_ply % 2 != 0:
        raise ValueError("n_ply must be even.")
    return len(candidate_angles) ** (n_ply // 2)


def enumerate_symmetric_layups(candidate_angles: List[float], n_ply: int) -> List[List[float]]:
    """
    All symmetric layup sequences (materialized).

    Prefer :func:`iterate_symmetric_layups` when n_ply is large: candidate count is
    ``len(candidate_angles) ** (n_ply // 2)`` and this list can exceed available RAM.
    """
    return list(iterate_symmetric_layups(candidate_angles, n_ply))


def _pool_process_count(n_jobs: int) -> int:
    if n_jobs == -1:
        return max(1, os.cpu_count() or 1)
    return max(1, int(n_jobs))


def _filter_layup_worker(args: tuple[List[float], bool]) -> List[float] | None:
    """Module-level worker for multiprocessing (must be picklable on Windows spawn)."""
    layup, strong_requirement = args
    requirements = EngineeringRequirements(strong_requirement=strong_requirement)
    return layup if requirements.filter(layup) else None


def parallel_filter_layups(
    layups: List[List[float]],
    requirements: EngineeringRequirements,
    n_jobs: int = -1,
    ) -> List[List[float]]:
    """
    Filter a list of layups in parallel (loads all ``layups`` into worker tasks).

    For many candidates, use :func:`parallel_filter_layups_streaming` instead.
    """
    n_proc = _pool_process_count(n_jobs)
    strong = requirements.strong_requirement
    work_items = [(layup, strong) for layup in layups]

    with multiprocessing.Pool(n_proc) as pool:
        filtered_layups = pool.map(_filter_layup_worker, work_items)

    return [layup for layup in filtered_layups if layup is not None]


def parallel_filter_layups_streaming(
    layups: Iterable[List[float]],
    strong_requirement: bool,
    n_jobs: int = -1,
    chunksize: int = 2048,
    ) -> List[List[float]]:
    """
    Filter layups in parallel without building a giant input list in the parent.

    The iterable is consumed incrementally; peak memory scales with chunksize and
    worker count, not with total candidate count.
    """
    n_proc = _pool_process_count(n_jobs)

    def work_gen() -> Iterator[tuple[List[float], bool]]:
        for layup in layups:
            yield (layup, strong_requirement)

    filtered: List[List[float]] = []
    with multiprocessing.Pool(n_proc) as pool:
        for result in pool.imap(_filter_layup_worker, work_gen(), chunksize=chunksize):
            if result is not None:
                filtered.append(result)
    return filtered


def build_or_extend_layup_database(
    n_ply_values: List[int] = list(range(8, 22, 2)),
    strong_requirement: bool = False,
    dataset_path: str | Path = DEFAULT_DATASET_PATH,
    old_database_path: str | Path = None,
    n_jobs: int = -1,
    filter_chunksize: int = 2048,
    ) -> pd.DataFrame:
    """
    Build or extend layup dataset with missing n_ply groups.

    Existing n_ply groups in the dataset are kept untouched, so the same file can
    be loaded and incrementally extended with larger/smaller n_ply sets later.

    Candidates are generated and filtered in a streaming fashion so large ``n_ply``
    does not require holding ``4**(n_ply/2)`` full stacking lists in memory at once.

    filter_chunksize
        ``imap`` chunk size for multiprocessing; larger values can be faster but use
        more RAM for in-flight tasks.
    """
    dataset_path = Path(dataset_path)
    dataset_path.parent.mkdir(parents=True, exist_ok=True)

    requirements = EngineeringRequirements(strong_requirement=strong_requirement)
    candidate_angles = requirements.candidate_angles

    source_path = old_database_path if old_database_path is not None else dataset_path
    current_df = load_layup_database(source_path)
    existing_n_ply = {int(n) for n in current_df["n_ply"].unique()} if not current_df.empty else set()
    missing_n_ply = [n for n in n_ply_values if n not in existing_n_ply]
    
    if not current_df.empty:
        print(f"Load existing layup database from [{source_path}] with {len(current_df)} rows")
        print(f"Existing n_ply groups: {sorted(existing_n_ply)}")
        print(f"Missing n_ply groups:  {sorted(missing_n_ply)}")
    else:
        print(f"Build new layup database from scratch")

    rows: list[dict[str, object]] = []
    for n_ply in missing_n_ply:
        
        print(f"Enumerate symmetric layups for n_ply={n_ply}")
        start_time = time.time()
        
        num_candidate_layups = symmetric_layup_candidate_count(candidate_angles, n_ply)
        new_layups = parallel_filter_layups_streaming(
            iterate_symmetric_layups(candidate_angles=candidate_angles, n_ply=n_ply),
            strong_requirement=requirements.strong_requirement,
            n_jobs=n_jobs,
            chunksize=filter_chunksize,
        )
        for i, layup in enumerate(new_layups):
            rows.append(
                {"n_ply": n_ply, "sub_id": i, "stacking": json.dumps(layup)}
            )
            
        elapsed_time = (time.time() - start_time) / 60.0
        print(f"Found {len(new_layups)} valid layup from {num_candidate_layups} candidates in {elapsed_time:.2f} min")

    new_df = pd.DataFrame(rows, columns=["n_ply", "sub_id", "stacking"])

    if current_df.empty and new_df.empty:
        out_df = pd.DataFrame(columns=["layup_id", "n_ply", "sub_id", "stacking"])
    elif current_df.empty:
        out_df = new_df.copy()
        out_df.insert(0, "layup_id", range(len(out_df)))
    elif new_df.empty:
        out_df = current_df.copy()
    else:
        base_id = int(current_df["layup_id"].max()) + 1
        new_df = new_df.copy()
        new_df.insert(0, "layup_id", range(base_id, base_id + len(new_df)))
        out_df = pd.concat([current_df, new_df], ignore_index=True)

    out_df["layup_id"] = out_df["layup_id"].astype(int)
    out_df["n_ply"] = out_df["n_ply"].astype(int)
    out_df["sub_id"] = out_df["sub_id"].astype(int)

    out_df = out_df.sort_values(["n_ply", "sub_id"], kind="stable").reset_index(drop=True)
    out_df.to_csv(dataset_path, index=False)
    return out_df


def _calculate_attributes_worker(layup: List[float]) -> Dict[str, int | List[float]]:
    """Module-level worker for multiprocessing (must be picklable on Windows spawn)."""
    results = Laminate.get_lamination_parameters(layup)
    results['xiA'] = np.round(results['xiA'], 6).tolist()
    results['xiB'] = np.round(results['xiB'], 6).tolist()
    results['xiD'] = np.round(results['xiD'], 6).tolist()
    results['n_90'] = int(layup.count(90))
    results['n_0'] = int(layup.count(0))
    return results


def calculate_attributes_for_layup_database(
        layup_database: pd.DataFrame,
        database_path: str = DEFAULT_DATASET_PATH,
        n_jobs: int = -1) -> pd.DataFrame:
    '''
    Calculate attributes for the layup database.
    
    Parameters
    ------------------
    layup_database: pd.DataFrame
        Layup database to calculate attributes for.
    database_path: str
        Path to save the updated layup database.
    n_jobs: int
        Number of jobs to run in parallel.
        If -1, use all available cores.
    
    Returns
    ------------------
    layup_database: pd.DataFrame
        Layup database with calculated attributes.
    '''
    n_proc = _pool_process_count(n_jobs)
    work_items = [json.loads(stacking) for stacking in layup_database["stacking"]]

    with multiprocessing.Pool(n_proc) as pool:
        results = pool.map(_calculate_attributes_worker, work_items)
    
    layup_database["xiA"] = [json.dumps(result['xiA']) for result in results]
    layup_database["xiB"] = [json.dumps(result['xiB']) for result in results]
    layup_database["xiD"] = [json.dumps(result['xiD']) for result in results]
    layup_database["n_90"] = [result['n_90'] for result in results]
    layup_database["n_0"] = [result['n_0'] for result in results]
    
    layup_database.to_csv(database_path, index=False)
    
    return layup_database


if __name__ == "__main__":
    
    # Example: build default dataset in data/layup_database.csv
    database = build_or_extend_layup_database(
        n_ply_values=list(range(8, 30, 2)),
        strong_requirement=False,
        dataset_path=DEFAULT_DATASET_PATH,
        old_database_path=os.path.join(DATA_PATH, "layup_database-old.csv"),
        n_jobs=32,
        filter_chunksize=int(2**20),
    )
    
    print(f"Saved layup database with {len(database)} rows to {DEFAULT_DATASET_PATH}")

    database_path = os.path.join(DATA_PATH, "layup_database-with-attributes.csv")
    calculate_attributes_for_layup_database(
        database,
        database_path=database_path,
        n_jobs=16,
    )
    print(f"Saved layup database with attributes to {database_path}")
    