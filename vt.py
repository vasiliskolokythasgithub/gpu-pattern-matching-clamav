#!/usr/bin/env python3
import os
import json
import subprocess
import glob
import time

API_KEY = "7ee3b28ab413342b1c2b4c9d256c6e8369b16834f46744be32483e632ca30a89"
results = []

for file_path in glob.glob("/tmp/test_corpus/by_type/office/*"):
    # Get SHA256
    sha = subprocess.check_output(["sha256sum", file_path]).decode().split()[0]
    
    # Query VirusTotal with rate limiting
    cmd = f"curl -s --request GET --url 'https://www.virustotal.com/api/v3/files/{sha}' --header 'x-apikey: {API_KEY}'"
    try:
        result = subprocess.check_output(cmd, shell=True, timeout=10)
        data = json.loads(result)
        
        # Check if response contains expected data
        if 'data' in data and 'attributes' in data['data']:
            stats = data['data']['attributes']['last_analysis_stats']
            malicious = stats['malicious']
            total = sum(stats.values())
            
            results.append({
                'file': os.path.basename(file_path),
                'sha256': sha,
                'detection': f"{malicious}/{total}",
                'malicious': malicious
            })
            print(f"✓ {os.path.basename(file_path)}: {malicious}/{total}")
        else:
            print(f"✗ {os.path.basename(file_path)}: Not found in VirusTotal database")
            
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError, json.JSONDecodeError) as e:
        print(f"✗ {os.path.basename(file_path)}: Error - {str(e)[:50]}")
    
     

# Sort by most detected
results.sort(key=lambda x: x['malicious'], reverse=True)

print("\n" + "="*50)
print("FINAL RESULTS")
print("="*50)
for r in results:
    print(f"{r['file']}: {r['detection']} engines detect this")

print(f"\nTotal files found in VirusTotal: {len(results)}/{len(glob.glob('/tmp/pe/*'))}")