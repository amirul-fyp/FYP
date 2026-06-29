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

# --- TELEGRAM NOTIFICATIONS ---
def send_telegram_alert(message):
    """Send a Telegram notification for critical threats."""
    bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
    chat_id = os.getenv('TELEGRAM_CHAT_ID')
    if not bot_token or not chat_id:
        print("⚠️ Telegram credentials not set – skipping alert.")
        return
    try:
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        payload = {
            "chat_id": chat_id,
            "text": message,
            "parse_mode": "HTML"
        }
        response = requests.post(url, json=payload, timeout=5)
        if response.status_code == 200:
            print("✅ Telegram alert sent!")
        else:
            print(f"⚠️ Telegram error: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"⚠️ Telegram send failed: {e}")

# --- GENERATIVE AI EXPLANATION (Groq) ---
def generate_ai_explanation(command, classification, confidence, risk_flags, entropy_score):
    """
    Use Groq's free API to generate a plain‑English explanation.
    Returns None if API key is missing, the call fails, or the response is empty.
    """
    api_key = os.getenv('GROQ_API_KEY')
    if not api_key:
        print("⚠️ Groq API key missing")
        return None

    # Detailed prompt to ensure a complete answer of at least 50 words
    prompt = f"""
You are a cybersecurity explainer. Explain this threat in simple, plain English.

Command: "{command}"
Threat type: {classification}
Confidence: {confidence}%
Risk flags: {risk_flags}
Complexity score: {entropy_score}

Provide a clear, complete explanation of at least 50 words. Use analogies if helpful. 
Do not include technical jargon. Ensure your explanation is fully self-contained and does not cut off early.
Only return the explanation text.
"""

    try:
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "llama-3.1-8b-instant",
            "messages": [
                {"role": "system", "content": "You are a helpful cybersecurity assistant."},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 600,   # Increased to allow longer explanations
            "temperature": 0.7
        }
        response = requests.post(url, headers=headers, json=payload, timeout=25)
        if response.status_code == 200:
            result = response.json()
            explanation = result['choices'][0]['message']['content'].strip()
            if not explanation or len(explanation) < 20:
                print(f"⚠️ Groq returned empty/short response: '{explanation}'")
                return None
            # No character cap – keep full explanation
            return explanation
        else:
            print(f"⚠️ Groq API error: {response.status_code} - {response.text}")
            return None
    except requests.exceptions.Timeout:
        print("⚠️ Groq API timeout")
        return None
    except Exception as e:
        print(f"⚠️ Groq error: {e}")
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
def process_command(command_input, attacker_ip="Unknown"):
    clean = re.sub(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', 'IP_ADDRESS', command_input)
    print(f"🛠️ Processing command: '{command_input}' from {attacker_ip}")

    if vectorizer is None or rf_model is None:
        print("❌ AI model not loaded – returning offline response")
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

        # Try generative AI first; if fails, use static fallback
        ai_explanation = generate_ai_explanation(command_input, best, best_prob, flags, entropy)
        if ai_explanation and ai_explanation.strip():
            explanation = ai_explanation
            print("✅ Used Groq AI explanation")
        else:
            explanation = generate_static_explanation(best, flags, entropy)
            print("ℹ️ Used static fallback explanation")

        reasoning = (
            f"🎯 <b>Confidence:</b> {best_prob:.1f}%|"
            f"🔑 <b>Key Triggers:</b> [{anchors}]|"
            f"💡 <b>Insight:</b> {insight}"
        )

        # --- SEND TELEGRAM ALERT FOR CRITICAL/HIGH THREATS ---
        print(f"🔔 Severity: {severity}, command: {command_input}")
        if severity in ["CRITICAL", "HIGH"]:
            alert_msg = (
                f"🚨 <b>CyberSentinel Alert</b>\n"
                f"📌 <b>Severity:</b> {severity}\n"
                f"🔹 <b>Attacker:</b> {attacker_ip}\n"
                f"🔹 <b>Command:</b> {command_input}\n"
                f"🔹 <b>Verdict:</b> {verdict}\n"
                f"🔹 <b>Confidence:</b> {best_prob:.1f}%\n"
                f"🔹 <b>Insight:</b> {insight}\n"
                f"🔹 <b>Time:</b> {datetime.datetime.now(MYT).strftime('%Y-%m-%d %H:%M:%S')} (MYT)"
            )
            send_telegram_alert(alert_msg)
        else:
            print(f"ℹ️ Severity {severity} – no Telegram alert")

        result = [{
            "event": "AI Analyzed Command",
            "details": command_input,
            "verdict": verdict,
            "reasoning": reasoning,
            "simple_explanation": explanation,
            "confidence": best_prob,
            "classification": best
        }]
        print(f"✅ Command processed: {verdict}")
        return result
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

# --- FLASK ROUTES ---
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
        stages = process_command(command, src_ip)
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
            print(f"📝 Appended log: {s['verdict']} for '{command}'")
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

@app.route('/api/get-logs')
def get_logs():
    db = get_db()
    response = jsonify(db)
    # Prevent caching
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

@app.route('/api/export-report')
def export():
    db = get_db()
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", size=10)

    pdf.set_font("Helvetica", 'B', 18)
    pdf.set_text_color(0, 51, 102)
    pdf.cell(0, 10, "CYBERSENTINEL FORENSIC REPORT", ln=True, align='C')
    pdf.set_font("Helvetica", 'I', 12)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 6, "AI-Enhanced Threat Intelligence", ln=True, align='C')
    pdf.ln(4)
    pdf.set_draw_color(0, 51, 102)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(6)

    pdf.set_font("Helvetica", size=10)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 6, f"Generated: {datetime.datetime.now(MYT).strftime('%Y-%m-%d %H:%M:%S')} (MYT)", ln=True)

    total = len(db)
    critical = sum(1 for a in db if "CRITICAL" in a.get('verdict', ''))
    high = sum(1 for a in db if "HIGH" in a.get('verdict', ''))
    warning = sum(1 for a in db if "WARNING" in a.get('verdict', ''))
    info = total - critical - high - warning

    pdf.ln(4)
    pdf.set_font("Helvetica", 'B', 11)
    pdf.set_text_color(0, 51, 102)
    pdf.cell(0, 6, "Summary", ln=True)
    pdf.set_font("Helvetica", size=10)
    pdf.set_text_color(0, 0, 0)

    for label, value in [("Total Events", total), ("Critical", critical), ("High", high), ("Warning", warning), ("Info", info)]:
        pdf.set_font("Helvetica", 'B', 10)
        pdf.cell(40, 6, label + ":", border=0)
        pdf.set_font("Helvetica", size=10)
        pdf.cell(20, 6, str(value), border=0, ln=True)

    pdf.ln(6)
    pdf.set_draw_color(200, 200, 200)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(4)

    if not db:
        pdf.set_font("Helvetica", 'I', 12)
        pdf.set_text_color(150, 150, 150)
        pdf.cell(0, 10, "No events recorded.", ln=True)
    else:
        for idx, a in enumerate(db, 1):
            if idx % 2 == 0:
                pdf.set_fill_color(240, 240, 240)
            else:
                pdf.set_fill_color(255, 255, 255)

            pdf.set_font("Helvetica", 'B', 10)
            pdf.set_text_color(0, 0, 0)
            pdf.cell(0, 6, f"Event #{idx}", ln=True, fill=True)

            for label, key in [("Timestamp:", "time"), ("Attacker IP:", "ip"), ("Target IP:", "target_ip"), ("Event:", "event"), ("Payload:", "details")]:
                pdf.set_font("Helvetica", 'B', 9)
                pdf.set_text_color(80, 80, 80)
                pdf.cell(30, 5, label, border=0, fill=True)
                pdf.set_font("Helvetica", size=9)
                pdf.set_text_color(0, 0, 0)
                val = a.get(key, '')[:100] if key == "details" else a.get(key, '')
                pdf.cell(0, 5, str(val), ln=True, fill=True)

            pdf.set_font("Helvetica", 'B', 9)
            pdf.set_text_color(80, 80, 80)
            pdf.cell(30, 5, "Verdict:", border=0, fill=True)
            verdict_text = a.get('verdict', '')
            if "CRITICAL" in verdict_text:
                pdf.set_text_color(200, 0, 0)
            elif "HIGH" in verdict_text:
                pdf.set_text_color(200, 100, 0)
            elif "WARNING" in verdict_text:
                pdf.set_text_color(200, 200, 0)
            else:
                pdf.set_text_color(0, 0, 0)
            pdf.set_font("Helvetica", 'B', 9)
            pdf.cell(0, 5, verdict_text, ln=True, fill=True)
            pdf.set_text_color(0, 0, 0)

            explanation = a.get('simple_explanation', 'No explanation available.')
            try:
                explanation = explanation.encode('latin-1', 'ignore').decode('latin-1')
            except:
                explanation = "Explanation unavailable."
            pdf.set_font("Helvetica", 'B', 9)
            pdf.set_text_color(80, 80, 80)
            pdf.cell(30, 5, "Explanation:", border=0, fill=True)
            pdf.set_font("Helvetica", size=9)
            pdf.set_text_color(0, 0, 0)
            pdf.multi_cell(0, 5, explanation, fill=True)

            pdf.ln(2)
            pdf.set_draw_color(200, 200, 200)
            pdf.line(10, pdf.get_y(), 200, pdf.get_y())
            pdf.ln(2)

    pdf.ln(10)
    pdf.set_y(-15)
    pdf.set_font("Helvetica", 'I', 8)
    pdf.set_text_color(150, 150, 150)
    pdf.cell(0, 10, f"Page {pdf.page_no()}", align='C')

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
        "groq_available": bool(os.getenv('GROQ_API_KEY')),
        "telegram_available": bool(os.getenv('TELEGRAM_BOT_TOKEN') and os.getenv('TELEGRAM_CHAT_ID'))
    })

@app.route('/test/telegram')
def test_telegram():
    message = "🚨 <b>Test Alert</b>\nThis is a test message from CyberSentinel."
    send_telegram_alert(message)
    return "Test alert sent! Check your Telegram."

# --- RENDER COMPATIBILITY ---
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)