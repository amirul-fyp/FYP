import json
import pandas as pd
import os

KAGGLE_FILE = 'cowrie.json'
OUTPUT_FILE = 'hacker_training_data.csv'

def parse_kaggle_logs():
    print(f"🔍 Scanning massive Kaggle log file: {KAGGLE_FILE}...")
    
    parsed_data = []
    
    try:
        with open(KAGGLE_FILE, 'r') as file:
            for line in file:
                try:
                    # Load each line as a JSON object
                    log_entry = json.loads(line.strip())
                    
                    # We ONLY care about the lines where a hacker typed a command
                    if log_entry.get('eventid') == 'cowrie.command.input':
                        command = log_entry.get('input', '').strip()
                        
                        if not command:
                            continue
                            
                        # --- AUTO-LABELING ENGINE (To build the Ground Truth) ---
                        cmd_lower = command.lower()
                        
                        if any(word in cmd_lower for word in ['wget', 'curl', 'xmrig', 'miner', 'coin']):
                            label = "Cryptojacking"
                        elif any(word in cmd_lower for word in ['ssh-rsa', 'crontab', 'useradd', 'chmod u+s']):
                            label = "Persistence"
                        elif any(word in cmd_lower for word in ['cat /etc', 'uname', 'lscpu', 'netstat', 'whoami']):
                            label = "Reconnaissance"
                        else:
                            label = "Normal"
                            
                        parsed_data.append({"command": command, "label": label})
                        
                except json.JSONDecodeError:
                    pass # Skip broken lines in the raw log
                    
    except FileNotFoundError:
        print(f"❌ ERROR: Could not find {KAGGLE_FILE} in this folder.")
        return

    # Convert to a clean spreadsheet
    df = pd.DataFrame(parsed_data)
    
    # Remove exact duplicates to keep the AI from over-memorizing the same line
    df = df.drop_duplicates()
    
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"✅ SUCCESS: Extracted {len(df)} unique attacker commands!")
    print(f"📁 Saved perfectly formatted data to {OUTPUT_FILE}")

if __name__ == "__main__":
    parse_kaggle_logs()