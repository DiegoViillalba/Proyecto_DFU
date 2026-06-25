#!/usr/bin/env python3
import os
import json
import requests
import time

# Paths
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
GDRIVE_JSON_PATH = "/home/diego-villalba/.gemini/antigravity-cli/brain/7bc12ab6-95b2-4f1d-8789-ef1279bb2905/scratch/gdrive_files.json"
DATA_DIR = os.path.join(REPO_ROOT, "data")

def download_file(url, dest_path, retries=3):
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    
    # Try downloading with retries and exponential backoff
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(url, stream=True, timeout=30)
            if r.status_code == 200:
                with open(dest_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
                return True
            elif r.status_code == 429:
                wait_time = attempt * 5
                print(f"Rate limited (429) for {os.path.basename(dest_path)}. Waiting {wait_time}s before retry {attempt}/{retries}...")
                time.sleep(wait_time)
            else:
                print(f"Failed to download {os.path.basename(dest_path)}: HTTP {r.status_code}. Retry {attempt}/{retries}...")
                time.sleep(attempt * 2)
        except Exception as e:
            print(f"Error downloading {os.path.basename(dest_path)}: {e}. Retry {attempt}/{retries}...")
            time.sleep(attempt * 2)
            
    return False

def main():
    if not os.path.exists(GDRIVE_JSON_PATH):
        print(f"Error: {GDRIVE_JSON_PATH} not found.")
        return

    with open(GDRIVE_JSON_PATH, 'r', encoding='utf-8') as f:
        files = json.load(f)

    print(f"Total files in Google Drive list: {len(files)}")
    
    # Separate list of files to download
    to_download = []
    skipped = 0
    
    for item in files:
        url = item['url']
        rel_path = item['path']
        dest_path = os.path.join(DATA_DIR, rel_path)
        
        # Check if it already exists and is non-empty
        # For dummy files in dfu_tissue, let's make sure we overwrite them!
        # If the file has 'dfu_tissue/train_images/img_' or 'dfu_tissue/val_images/img_', it's a dummy file, we must replace it.
        # Otherwise, if it already exists, we skip it.
        is_dummy = ("dfu_tissue/train_images/img_" in rel_path or 
                    "dfu_tissue/train_masks/img_" in rel_path or
                    "dfu_tissue/val_images/img_" in rel_path or
                    "dfu_tissue/val_masks/img_" in rel_path)
        
        if os.path.exists(dest_path) and os.path.getsize(dest_path) > 0 and not is_dummy:
            skipped += 1
        else:
            to_download.append((url, dest_path, rel_path))

    print(f"Skipping {skipped} files that are already present.")
    print(f"Need to download {len(to_download)} files.")
    
    if len(to_download) == 0:
        print("All files are already downloaded!")
        return

    success_count = 0
    fail_count = 0
    
    for i, (url, dest_path, rel_path) in enumerate(to_download):
        print(f"[{i+1}/{len(to_download)}] Downloading {rel_path}...")
        
        # Download
        success = download_file(url, dest_path)
        if success:
            success_count += 1
            # Add a small delay between requests to be polite to Google Drive API
            time.sleep(0.2)
        else:
            print(f"  ERROR: Failed to download {rel_path} after retries.")
            fail_count += 1
            
    print(f"\nDownload finished. Success: {success_count}, Failed: {fail_count}")

if __name__ == '__main__':
    main()
