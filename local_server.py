from flask import Flask, request, jsonify, render_template, send_file
import datetime
import os
import re
import math
import base64
import json
from fpdf import FPDF
import joblib
import numpy as np
import pickle
import warnings
import traceback
import requests

# Suppress warnings
warnings.filterwarnings("ignore")

# Lock the server to Malaysian Time (UTC+8)
MYT = datetime.timezone(datetime.timedelta(hours=8))

app = Flask(__name__)

# Cloud-Safe Path Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPORT_FOLDER = os.path.join(BASE_DIR, "SUMMARY_REPORT")
DB_FILE = os.path.join(BASE_DIR, "attack_database.json")

if not os.path.exists(REPORT_FOLDER):
    os.makedirs(REPORT_FOLDER)

# --- DATABASE HELPERS ---
def get_db():
    if os.path.exists(DB_FILE):
        try:
            with open(DB_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"⚠️ Database read error: {e}")
            return []
    return []

def save_db(data):
    try:
        with open(DB_FILE, 'w') as f:
            json.dump(data, f, indent=4)
    except Exception as e:
        print(f"⚠️ Database write error: {e}")

# --- LOAD THE MACHINE LEARNING MODEL ---
print("=" * 60)
print("🚀 Starting CyberSentinel Server...")
print("=" * 60)

vectorizer = None
rf_model = None
model_load_error = None

model_path = os.path.join(BASE_DIR, "text_translator.pkl")
rf_path = os.path.join(BASE_DIR, "random_forest_model.pkl")

try:
    import sklearn
    print(f"📦 scikit-learn version: {sklearn.__version__}")
except ImportError:
    print("❌ scikit-learn is not installed.")
    model_load_error = "scikit-learn missing"

if os.path.exists(model_path) and os.path.exists(rf_path):
    try:
        print("🧠 Loading with joblib...")
        vectorizer = joblib.load(model_path)
        rf_model = joblib.load(rf_path)
        print("✅ Loaded successfully with joblib.")
    except ModuleNotFoundError as e:
        if "numpy._core" in str(e):
            print("⚠️ NumPy version mismatch. Trying pickle fallback...")
            try:
                with open(model_path, 'rb') as f:
                    vectorizer = pickle.load(f)
                with open(rf_path, 'rb') as f:
                    rf_model = pickle.load(f)
                print("✅ Loaded with pickle fallback!")
            except Exception as p_e:
                model_load_error = f"Pickle failed: {p_e}"
                print(f"❌ {model_load_error}")
        else:
            model_load_error = str(e)
            print(f"❌ Joblib error: {model_load_error}")
    except Exception as e:
        model_load_error = str(e)
        print(f"❌ General load error: {model_load_error}")
else:
    model_load_error = "Model files not found. Please upload random_forest_model.pkl and text_translator.pkl"
    print(f"❌ {model_load_error}")

if vectorizer is not None and rf_model is not None:
    print("✅ Random Forest AI is ready!")
    if hasattr(rf_model, 'classes_'):
        print(f"📊 Classes: {rf_model.classes_}")
else:
    print("⚠️ Running in offline mode without AI detection.")

print("=" * 60)

# --- GENERATIVE AI EXPLANATION (Hugging Face Free Inference) ---
def generate_ai_explanation(command, classification, confidence, risk_flags, entropy_score):
    """Use Hugging Face free inference API to generate plain‑English explanation."""
    api_key = os.getenv('HF_API_TOKEN')
    if not api_key:
        return None

    # Format prompt for Mistral‑7B Instruct
    prompt = f"""
<|user|>
You are a cybersecurity explainer. Explain this threat in simple, plain English.

Command: "{command}"
Threat type: {classification}
Confidence: {confidence}%
Risk flags: {risk_flags}
Complexity score: {entropy_score}

Write a 2‑3 sentence explanation that a non‑technical person can understand.
Be friendly, use analogies if helpful, and do not include technical jargon.
Only return the explanation text.
<|assistant|>
"""

    try:
        API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.1"
        headers = {"Authorization": f"Bearer {api_key}"}
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 150,
                "temperature": 0.7,
                "return_full_text": False
            }
        }
        response = requests.post(API_URL, headers=headers, json=payload, timeout=10)
        
        if response.status_code == 200:
            result = response.json()
            # Hugging Face returns a list with generated text
            if isinstance(result, list) and len(result) > 0:
                explanation = result[0].get('generated_text', '').strip()
                # If the response starts with the prompt, strip it off
                if explanation.startswith("<|assistant|>"):
                    explanation = explanation.replace("<|assistant|>", "").strip()
                if len(explanation) > 300:
                    explanation = explanation[:297] + "..."
                return explanation
            else:
                print(f"⚠️ Unexpected Hugging Face response format: {result}")
                return None
        else:
            print(f"⚠️ Hugging Face API error: {response.status_code} - {response.text}")
            return None
    except requests.exceptions.Timeout:
        print("⚠️ Hugging Face API timeout – using fallback.")
        return None
    except Exception as e:
        print(f"⚠️ Hugging Face error: {e}")
        return None

# --- STATIC FALLBACK EXPLANATION ---
def generate_static_explanation(classification, risk_flags, entropy_score):
    explanations = {
        "Cryptojacking": "🚨 This looks like someone trying to use your computer to mine cryptocurrency without permission. It's like someone secretly using your electricity to run their Bitcoin machine!",
        "Persistence": "🔐 This seems like someone trying to install a hidden backdoor to keep accessing your system later. Think of it as someone leaving a spare key under the mat.",
        "Reconnaissance": "🔍 This appears to be an attacker checking out your system, like a burglar casing a house before breaking in.",
        "Routine Noise": "✅ This looks like normal system activity. It's like background noise in a busy office.",
        "Unknown": "⚠️ Unusual activity detected on your system. Further investigation may be needed."
    }
    base = explanations.get(classification, explanations["Unknown"])
    if risk_flags:
        risk_text = " and ".join(risk_flags) if len(risk_flags) > 1 else risk_flags[0]
        base += f" The specific risky commands detected were: {risk_text}."
    if entropy_score > 0.65:
        base += " The command looks complex and obfuscated, which is often a sign of malicious intent."
    return base

# --- CORE PROCESSING FUNCTION ---
def process_command(command_input):
    clean = re.sub(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', 'IP_ADDRESS', command_input)
    if vectorizer is None or rf_model is None:
        return [{
            "event": "AI Offline",
            "details": command_input,
            "verdict": "INFO: AI Offline",
            "reasoning": f"ML model not loaded: {model_load_error}",
            "simple_explanation": "⚠️ AI detection is offline. Please check server logs.",
            "confidence": 0,
            "classification": "Unknown"
        }]
    try:
        features = vectorizer.transform([clean])
        proba = rf_model.predict_proba(features)[0]
        classes = rf_model.classes_
        if len(classes) == 1:
            best = classes[0]
            best_prob = 100.0
            second = classes[0]
            second_prob = 0.0
        else:
            idx = np.argsort(proba)[::-1]
            best = classes[idx[0]]
            best_prob = proba[idx[0]] * 100
            second = classes[idx[1]]
            second_prob = proba[idx[1]] * 100

        feature_names = vectorizer.get_feature_names_out()
        weights = features.toarray()[0]
        top_idx = weights.argsort()[-3:][::-1]
        top_words = [feature_names[i] for i in top_idx if weights[i] > 0]

        entropy = 0.0
        entropy_status = "Standard"
        if len(clean) > 1:
            for ch in set(clean):
                p = clean.count(ch) / len(clean)
                entropy -= p * math.log(p, 2)
            max_ent = math.log(min(len(clean), 256), 2)
            norm = entropy / max_ent if max_ent > 0 else 0
            if norm > 0.80 and len(clean) > 12:
                entropy_status = "High Risk"
            elif norm > 0.65:
                entropy_status = "Elevated"
        entropy = round(entropy, 2)

        risk_keywords = ['curl', 'wget', 'bash', 'sh', 'chmod', 'nc -e', 'crontab']
        flags = [kw for kw in risk_keywords if kw in command_input.lower()]
        if 'wget' in command_input.lower() or 'curl' in command_input.lower():
            best = "Cryptojacking"

        map_sev = {
            "Cryptojacking": ("CRITICAL", "High-resource consumption detected."),
            "Persistence": ("HIGH", "Potential backdoor installation detected."),
            "Reconnaissance": ("WARNING", "Information gathering detected."),
            "Routine Noise": ("INFO", "Standard system operation.")
        }
        severity, insight = map_sev.get(best, ("INFO", "Uncategorized."))
        verdict = f"{severity}: AI Detected {best}"

        anchors = ", ".join([f"'{w}'" for w in top_words]) if top_words else "none"

        # Try generative AI first
        ai_explanation = generate_ai_explanation(command_input, best, best_prob, flags, entropy)
        if ai_explanation:
            explanation = ai_explanation
        else:
            explanation = generate_static_explanation(best, flags, entropy)

        reasoning = (
            f"🎯 <b>Confidence:</b> {best_prob:.1f}%|"
            f"🔑 <b>Key Triggers:</b> [{anchors}]|"
            f"💡 <b>Insight:</b> {insight}"
        )

        return [{
            "event": "AI Analyzed Command",
            "details": command_input,
            "verdict": verdict,
            "reasoning": reasoning,
            "simple_explanation": explanation,
            "confidence": best_prob,
            "classification": best
        }]
    except Exception as e:
        print(f"⚠️ Processing error: {e}")
        traceback.print_exc()
        return [{
            "event": "AI Processing Error",
            "details": command_input,
            "verdict": "INFO: AI Error",
            "reasoning": f"Error: {str(e)}",
            "simple_explanation": f"⚠️ Processing error: {str(e)}",
            "confidence": 0,
            "classification": "Unknown"
        }]

# --- FLASK ROUTES (unchanged) ---
@app.route('/')
def index():
    db = get_db()
    try:
        return render_template('dashboard.html', alerts=db)
    except Exception as e:
        return "Dashboard template not found. Please create templates/dashboard.html", 404

@app.route('/api/logs', methods=['POST'])
def receive_logs():
    data = request.json
    db = get_db()
    src_ip = data.get("attacker_ip", data.get("src_ip", "Unknown IP"))
    target_ip = data.get("target_ip", "Unknown Target IP")
    if isinstance(target_ip, list) and target_ip:
        target_ip = target_ip[0]

    if "command" in data:
        event_type = "cowrie.command.input"
        command = data.get("command", "")
    else:
        event_type = data.get("eventid", "unknown")
        command = data.get("input", "")

    if data.get("is_encoded"):
        try:
            command = base64.b64decode(command.encode()).decode()
        except Exception as e:
            print(f"⚠️ Base64 decode error: {e}")

    if event_type == "cowrie.command.input":
        stages = process_command(command)
        for s in stages:
            db.append({
                "time": datetime.datetime.now(MYT).strftime("%Y-%m-%d %H:%M:%S"),
                "ip": src_ip,
                "attacker_ip": src_ip,
                "target_ip": target_ip,
                "event": s['event'],
                "details": s['details'],
                "verdict": s['verdict'],
                "reasoning": s.get('reasoning', ''),
                "simple_explanation": s.get('simple_explanation', ''),
                "confidence": s.get('confidence', 0),
                "classification": s.get('classification', 'Unknown')
            })
    else:
        details = str(data.get("message", "N/A"))
        verdict = "INFO: General Event"
        if event_type == "cowrie.login.failed":
            details = f"User: {data.get('username')} | Pass: {data.get('password')}"
            verdict = "HIGH: SSH Brute Force Attempt"
        elif event_type == "cowrie.session.file_download":
            details = data.get("url", "Unknown URL")
            verdict = "CRITICAL: Malware Source Download"
        db.append({
            "time": datetime.datetime.now(MYT).strftime("%Y-%m-%d %H:%M:%S"),
            "ip": src_ip,
            "attacker_ip": src_ip,
            "target_ip": target_ip,
            "event": event_type,
            "details": details,
            "verdict": verdict,
            "reasoning": "Standard event.",
            "simple_explanation": "General system event.",
            "confidence": 0,
            "classification": "General Event"
        })
    save_db(db)
    return jsonify({"status": "received"}), 200

@app.route('/api/analyze')
def analyze():
    db = get_db()
    total = len(db)
    critical = sum(1 for a in db if "CRITICAL" in a.get('verdict', ''))
    return jsonify({"total_attacks": total, "critical_threats": critical})

@app.route('/api/clear', methods=['POST'])
def clear():
    save_db([])
    return jsonify({"status": "success"})

@app.route('/api/chart-data')
def chart():
    db = get_db()
    counts = {"CRITICAL": 0, "HIGH": 0, "WARNING": 0, "INFO": 0}
    ips = set()
    for a in db:
        v = a.get('verdict', 'INFO')
        ip = a.get('ip', '')
        if ip and ip != "Unknown IP":
            ips.add(ip)
        if "CRITICAL" in v:
            counts["CRITICAL"] += 1
        elif "HIGH" in v:
            counts["HIGH"] += 1
        elif "WARNING" in v:
            counts["WARNING"] += 1
        else:
            counts["INFO"] += 1
    total = sum(counts.values())
    if total == 0:
        html = "<span class='text-info'>✅ System secure. Awaiting activity.</span>"
        border = "border-info"
    elif counts["CRITICAL"] > 0:
        html = f"<strong class='text-danger'>🚨 CRITICAL: {', '.join(ips)}</strong>"
        border = "border-danger"
    else:
        html = "<span class='text-success'>✅ Routine monitoring.</span>"
        border = "border-success"
    return jsonify({"counts": counts, "insight_html": html, "border_class": border})

@app.route('/api/export-report')
def export():
    db = get_db()
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    def add_line(txt, style='', size=10, align='L'):
        pdf.set_font("Courier", style=style, size=size)
        clean = txt.encode('ascii', 'ignore').decode('ascii')
        pdf.multi_cell(0, 6, clean, align=align)

    add_line("=" * 60, style='B')
    add_line("CYBERSENTINEL FORENSIC REPORT", style='B', size=12, align='C')
    add_line("=" * 60)
    add_line(f"Generated: {datetime.datetime.now(MYT).strftime('%Y-%m-%d %H:%M:%S')} (MYT)")
    add_line("")
    if not db:
        add_line("No events.")
    else:
        for i, a in enumerate(db, 1):
            ip = a.get('ip', 'Unknown')
            add_line(f"{i}. [{a['time']}] {ip} -> {a.get('target_ip', 'N/A')}")
            add_line(f"   Event: {a['event']}")
            add_line(f"   Payload: {a['details'][:100]}")
            add_line(f"   Verdict: {a['verdict']}")
            add_line(f"   Explanation: {a.get('simple_explanation', 'N/A')}")
            add_line("   ---")
    filename = f"Report_{datetime.datetime.now(MYT).strftime('%Y%m%d_%H%M%S')}.pdf"
    path = os.path.join(REPORT_FOLDER, filename)
    pdf.output(path)
    return send_file(path, as_attachment=True, download_name=filename)

@app.route('/api/health')
def health():
    return jsonify({
        "status": "running",
        "timestamp": datetime.datetime.now(MYT).strftime("%Y-%m-%d %H:%M:%S"),
        "random_forest_loaded": vectorizer is not None and rf_model is not None,
        "model_files_exist": os.path.exists(model_path) and os.path.exists(rf_path),
        "load_error": model_load_error,
        "database_size": len(get_db()),
        "hf_available": bool(os.getenv('HF_API_TOKEN'))
    })

# --- RENDER COMPATIBILITY ---
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)"# Trigger rebuild" 
