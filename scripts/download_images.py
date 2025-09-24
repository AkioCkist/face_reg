#!/usr/bin/env python3
"""Download images from a templated URL for a range of IDs.

Example:
  python scripts/download_images.py --start 22040000 --end 22040020

By default it will fetch URLs like:
  http://staff.vnuk.edu.vn:5000/static/captures/{id}.jpg

The script supports zero-padding the id, concurrent downloads, retries and logging.
"""
import os
import sys
import argparse
import logging
from pathlib import Path
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# Ensure project root is importable (so setup_logging and other project modules resolve)
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

try:
    import requests
except Exception:
    requests = None

from setup_logging import setup_logging


logger, _ = setup_logging("download_images", logging.INFO)


def download_one(id_val: int, template: str, outdir: Path, pad: int, timeout: float, retries: int, delay: float, overwrite: bool = False):
    id_str = str(id_val).zfill(pad) if pad and pad > 0 else str(id_val)
    url = template.format(id=id_str)
    outpath = outdir / f"{id_str}.jpg"

    # Skip if file exists
    if outpath.exists():
        if overwrite:
            try:
                outpath.unlink()
            except Exception:
                pass
        else:
            return (id_val, True, "exists")

    if requests is None:
        return (id_val, False, "requests_not_installed")

    attempt = 0
    while attempt <= retries:
        try:
            resp = requests.get(url, timeout=timeout, stream=True)
            if resp.status_code == 200:
                # Write to disk and ensure data is flushed to storage immediately
                try:
                    with open(outpath, "wb") as f:
                        for chunk in resp.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                        try:
                            f.flush()
                            os.fsync(f.fileno())
                        except Exception:
                            # If fsync not available or fails, ignore but file was written
                            pass
                    return (id_val, True, "downloaded")
                except Exception as e:
                    reason = str(e)
                    logger.debug(f"ID {id_str}: Write failed {e}")
            else:
                reason = f"http_{resp.status_code}"
                logger.debug(f"ID {id_str}: HTTP {resp.status_code} for {url}")
        except Exception as e:
            reason = str(e)
            logger.debug(f"ID {id_str}: Exception {e}")

        attempt += 1
        if attempt <= retries:
            time.sleep(delay)

    return (id_val, False, reason)


def main():
    p = argparse.ArgumentParser(description="Batch download images by ID range")
    p.add_argument("--start", type=int, required=False, default=None, help="Start ID (inclusive)")
    p.add_argument("--end", type=int, required=False, default=None, help="End ID (inclusive)")
    p.add_argument("--template", default="http://staff.vnuk.edu.vn:5000/static/captures/{id}.jpg", help="URL template where {id} is replaced")
    p.add_argument("--pad", type=int, default=0, help="Zero-pad ID width, e.g. 8 for 00001234")
    p.add_argument("--outdir", default="downloads", help="Output directory to save images")
    p.add_argument("--ids", default=None, help="Python-style list string of ids, e.g. '[23060016,23060006,...]'")
    p.add_argument("--workers", type=int, default=8, help="Concurrent download workers")
    p.add_argument("--timeout", type=float, default=8.0, help="HTTP request timeout seconds")
    p.add_argument("--retries", type=int, default=2, help="Number of retries per ID on failure")
    p.add_argument("--delay", type=float, default=0.5, help="Delay between retries (seconds)")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing files")
    args = p.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    start = args.start
    end = args.end
    # If --ids provided, parse that instead of using start/end range
    if args.ids:
        try:
            import ast
            parsed = ast.literal_eval(args.ids)
            if not isinstance(parsed, (list, tuple)):
                logger.error("--ids must evaluate to a list or tuple of IDs")
                sys.exit(2)
            ids = [int(x) for x in parsed]
        except Exception as e:
            logger.error(f"Failed to parse --ids: {e}")
            sys.exit(2)
    else:
        # Require start and end when --ids not provided
        if start is None or end is None:
            logger.error("Either --ids or both --start and --end must be provided")
            sys.exit(2)

        if end < start:
            logger.error("End ID must be >= start ID")
            sys.exit(2)

        ids = list(range(start, end + 1))
    total = len(ids)
    logger.info(f"Downloading {total} images to {outdir} using template {args.template}")

    # If overwrite not set, skip files that exist (download_one handles that)
    results = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        future_to_id = {ex.submit(download_one, i, args.template, outdir, args.pad, args.timeout, args.retries, args.delay, args.overwrite): i for i in ids}
        for fut in as_completed(future_to_id):
            i = future_to_id[fut]
            try:
                id_val, ok, reason = fut.result()
            except Exception as e:
                logger.error(f"ID {i}: worker raised {e}")
                results.append((i, False, str(e)))
            else:
                results.append((id_val, ok, reason))
                if ok and reason == "downloaded":
                    logger.info(f"Downloaded {id_val}")
                elif ok and reason == "exists":
                    logger.debug(f"Skipped {id_val} (exists)")
                else:
                    logger.debug(f"Failed {id_val}: {reason}")

    # Summary
    succ = [r for r in results if r[1]]
    fail = [r for r in results if not r[1]]
    logger.info(f"Finished. Success: {len(succ)}; Failed: {len(fail)}")
    if fail:
        logger.info("Failures (first 20):")
        for item in fail[:20]:
            logger.info(f"  ID {item[0]} -> {item[2]}")


if __name__ == "__main__":
    main()
