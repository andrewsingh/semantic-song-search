#!/usr/bin/env python3

import json
import argparse
from pathlib import Path
from typing import List
from pydantic import BaseModel, ValidationError
import sys

sys.path.append(str(Path(__file__).parent.parent))
from profiles import SongProfile



def is_successful_entry(entry: dict) -> bool:
    """
    Check if an entry is successful based on:
    1. Matches SongProfile schema
    2. familiar == True
    3. No null values
    """
    try:
        # Validate against schema
        profile = SongProfile(**entry)
        
        # Check if familiar is True
        if not profile.familiar:
            return False
        
        # Check for null values (None in Python)
        for field_name, field_value in profile.model_dump().items():
            if field_value is None:
                return False
        
        return True
        
    except ValidationError:
        return False
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Filter song profiles into successful and failed buckets"
    )
    parser.add_argument(
        "-i", "--input", 
        required=True,
        help="Path to input JSONL file containing song profiles"
    )
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    
    if not input_path.exists():
        print(f"Error: Input file '{input_path}' does not exist")
        return
    
    # Generate output file names
    stem = input_path.stem  # filename without extension
    parent = input_path.parent
    success_path = parent / f"{stem}_success.jsonl"
    failed_path = parent / f"{stem}_failed.jsonl"
    
    successful_entries = []
    failed_entries = []
    total_entries = 0
    
    print(f"Reading entries from: {input_path}")
    
    # Read and process entries
    with open(input_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
                
            total_entries += 1
            
            try:
                entry = json.loads(line)
                
                if is_successful_entry(entry):
                    successful_entries.append(entry)
                else:
                    failed_entries.append(entry)
                    
            except json.JSONDecodeError as e:
                print(f"Warning: Skipping malformed JSON on line {line_num}: {e}")
                # Treat malformed JSON as failed entry
                failed_entries.append({"error": f"Malformed JSON on line {line_num}", "raw_line": line})
    
    # Write successful entries
    print(f"Writing {len(successful_entries)} successful entries to: {success_path}")
    with open(success_path, 'w', encoding='utf-8') as f:
        for entry in successful_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    # Write failed entries
    print(f"Writing {len(failed_entries)} failed entries to: {failed_path}")
    with open(failed_path, 'w', encoding='utf-8') as f:
        for entry in failed_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    # Summary
    success_rate = (len(successful_entries) / total_entries * 100) if total_entries > 0 else 0
    
    print(f"\n{'='*50}")
    print(f"FILTERING COMPLETE")
    print(f"{'='*50}")
    print(f"Total entries processed: {total_entries}")
    print(f"  ├─ Successful: {len(successful_entries)} ({success_rate:.1f}%)")
    print(f"  └─ Failed: {len(failed_entries)} ({100-success_rate:.1f}%)")
    print(f"Successful entries saved to: {success_path}")
    print(f"Failed entries saved to: {failed_path}")


if __name__ == "__main__":
    main() 