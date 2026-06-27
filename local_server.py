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
import io
import warnings
import pickle

import google.generativeai as genai

# ==========================================
# 🛑 PUT YOUR GOOGLE AI API KEY HERE 🛑
# ==========================================
genai.configure(api_key="AQ.Ab8RN6LybSnZjOhTbCXJjWzNBeiirEeHDXQw7vIPOnmyJoXndA")

# Lock the server to Malaysian Time (UTC+8)
MYT = datetime.timezone(datetime.timedelta(hours=8))

app = Flask(__name__)

# Cloud-Safe Path Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPORT_FOLDER = os.path.join(BASE_DIR, "SUMMARY_REPORT")
DB_FILE = os.path.join(BASE_DIR, "attack_database.json")

if not os.path.exists(REPORT_FOLDER):
    os.makedirs(REPORT_FOLDER)

# --- CLOUD DATABASE HELPER FUNCTIONS ---
def get_db():
    if os.path.exists(DB_FILE):
        try:
            with open(DB_FILE, 'r') as f:
                return json.load(f)
        except Exception:
            return []
    return []

def save_db(data):
    with open(DB_FILE, 'w') as f:
        json.dump(data, f, indent=4)

# --- LOAD THE MACHINE LEARNING MODEL ---
# --- LOAD THE MACHINE LEARNING MODEL ---
ml_error_message = "Unknown Error"
vectorizer = None
rf_model = None

try:
    print("🧠 Waking up Random Forest AI...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        vectorizer = joblib.load(os.path.join(BASE_DIR, "text_translator.pkl"))
        rf_model = joblib.load(os.path.join(BASE_DIR, "random_forest_model.pkl"))
    print("✅ AI Loaded Successfully!")
    ml_error_message = None
except ModuleNotFoundError as e:
    if "numpy._core" in str(e):
        print("⚠️ NumPy version mismatch. Attempting pickle fallback...")
        try:
            import pickle
            with open(os.path.join(BASE_DIR, "text_translator.pkl"), 'rb') as f:
                vectorizer = pickle.load(f)
            with open(os.path.join(BASE_DIR, "random_forest_model.pkl"), 'rb') as f:
                rf_model = pickle.load(f)
            print("✅ Loaded with pickle fallback!")
            ml_error_message = None
        except Exception as p_e:
            ml_error_message = f"Pickle fallback also failed: {p_e}"
            vectorizer = None
            rf_model = None
    else:
        ml_error_message = str(e)
        vectorizer = None
        rf_model = None
except Exception as e:
    ml_error_message = str(e)
    print(f"❌ AI Loading Failed. Error: {ml_error_message}")
    vectorizer = None
    rf_model = None

# --- AI-DRIVEN FORENSIC ENGINE ---
def process_command(command_input):
    stages = []

    clean_input = re.sub(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', 'IP_ADDRESS', command_input)

    # Check if AI models are loaded (Outputs the specific error to the dashboard)
    if vectorizer is None or rf_model is None:
        return [{
            "event": "AI Offline",
            "details": command_input,
            "verdict": "INFO: AI Offline",
            "reasoning": f"ML Model Load Failure: {ml_error_message}",
            "advanced": "N/A"
        }]

    try:
        math_features = vectorizer.transform([clean_input])
        probabilities = rf_model.predict_proba(math_features)[0]
        classes = rf_model.classes_

        if len(classes) == 1:
            best_class = classes[0]
            best_prob = 100.0
            second_class = classes[0]
            second_prob = 0.0
        else:
            sorted_indices = np.argsort(probabilities)[::-1]
            best_class = classes[sorted_indices[0]]
            best_prob = probabilities[sorted_indices[0]] * 100
            second_class = classes[sorted_indices[1]]
            second_prob = probabilities[sorted_indices[1]] * 100

        feature_names = vectorizer.get_feature_names_out()
        weights = math_features.toarray()[0]
        important_indices = weights.argsort()[-3:][::-1]
        influential_words = [feature_names[i] for i in important_indices if weights[i] > 0]

        # PLAIN ENGLISH ENTROPY EXPLANATION
        entropy = 0
        entropy_status = "Standard"
        entropy_desc = "Reads like normal, typed human commands."
        if len(clean_input) > 1:
            for x in set(clean_input):
                p_x = float(clean_input.count(x)) / len(clean_input)
                entropy += -p_x * math.log(p_x, 2)
            max_entropy = math.log(min(len(clean_input), 256), 2)
            normalized = entropy / max_entropy if max_entropy > 0 else 0

            if normalized > 0.80 and len(clean_input) > 12:
                entropy_status = "High Risk"
                entropy_desc = "Looks heavily scrambled, encoded, or obfuscated (e.g., Base64)."
            elif normalized > 0.65:
                entropy_status = "Elevated"
                entropy_desc = "Contains complex syntax, often used in automated scripts."
        entropy = round(entropy, 2)

        high_risk_keywords = ['curl', 'wget', 'bash', 'sh', 'chmod', 'nc -e', 'crontab']
        risk_flags = [kw for kw in high_risk_keywords if kw in command_input.lower()]

        if 'wget' in command_input.lower() or 'curl' in command_input.lower():
            best_class = "Cryptojacking"

        narrative_map = {
            "Cryptojacking": ("CRITICAL", "High-resource consumption detected. Potential for unauthorized pool connection and crypto-mining execution."),
            "Persistence": ("HIGH", "Attempt to modify system state for long-term access. Investigating potential backdoor installation."),
            "Reconnaissance": ("WARNING", "Information gathering detected. Attacker is mapping file system architecture and user environment."),
            "Routine Noise": ("INFO", "Standard system operation. No indicators of compromise detected.")
        }

        severity, insight = narrative_map.get(best_class, ("INFO", "Uncategorized command execution."))
        verdict = f"{severity}: AI Detected {best_class}"

        anchors = ", ".join([f"'{w}'" for w in influential_words]) if influential_words else "none"

        ai_reasoning = (
            f"🎯 <b>Confidence Score:</b> {best_prob:.1f}% certainty.|"
            f"🔑 <b>Key Triggers:</b> The AI flagged these specific words: [{anchors}].|"
            f"💡 <b>Analyst Insight:</b> {insight}"
        )

        risk_flags_str = ', '.join(risk_flags) if risk_flags else 'No manual risk keywords found.'
        advanced_analysis = (
            f"🤖 <b>Alternative Guess:</b> {second_prob:.1f}% chance this is actually '{second_class}'.|"
            f"🚩 <b>Hardcoded Rules:</b> {risk_flags_str}|"
            f"🕵️ <b>Obfuscation Check:</b> {entropy_status} (Score {entropy}). {entropy_desc}"
        )

        stages.append({
            "event": "AI Analyzed Command",
            "details": command_input,
            "verdict": verdict,
            "reasoning": ai_reasoning,
            "advanced": advanced_analysis
        })

    except Exception as e:
        print(f"⚠️ Error in AI processing: {e}")
        stages.append({
            "event": "AI Processing Error",
            "details": command_input,
            "verdict": "INFO: AI Error",
            "reasoning": f"Error occurred: {str(e)}",
            "advanced": "N/A"
        })

    return stages

# --- FLASK ROUTES ---
@app.route('/')
def index():
    attack_database = get_db()
    return render_template('dashboard.html', alerts=attack_database)

@app.route('/api/logs', methods=['POST'])
def receive_logs():
    log_data = request.json
    attack_database = get_db()

    src_ip = log_data.get("attacker_ip", log_data.get("src_ip", "Unknown IP"))
    if not src_ip or str(src_ip).strip() == "":
        src_ip = "Unknown IP"

    target_ip = log_data.get("target_ip", "Unknown Target IP")
    if isinstance(target_ip, list) and len(target_ip) > 0:
        target_ip = target_ip[0]

    if "command" in log_data:
        event_type = "cowrie.command.input"
        command_input = log_data.get("command", "")
    else:
        event_type = log_data.get("eventid", "unknown")
        command_input = log_data.get("input", "")

    if log_data.get("is_encoded"):
        try:
            command_input = base64.b64decode(command_input.encode('utf-8')).decode('utf-8')
        except Exception as e:
            print(f"⚠️ Base64 Decode Error: {e}")

    if event_type == "cowrie.command.input":
        stages = process_command(command_input)
        for stage in stages:
            attack_database.append({
                "time": datetime.datetime.now(MYT).strftime("%Y-%m-%d %H:%M:%S"),
                "ip": src_ip,
                "attacker_ip": src_ip,
                "target_ip": target_ip,
                "event": stage['event'],
                "details": stage['details'],
                "verdict": stage['verdict'],
                "reasoning": stage.get('reasoning', "Model reasoning unavailable."),
                "advanced": stage.get('advanced', "N/A")
            })
    else:
        details = str(log_data.get("message", "N/A"))
        verdict = "INFO: General Event"

        if event_type == "cowrie.login.failed":
            details = f"User: {log_data.get('username')} | Pass: {log_data.get('password')}"
            verdict = "HIGH: SSH Brute Force Attempt"
        elif event_type == "cowrie.session.file_download":
            details = log_data.get("url", "Unknown URL")
            verdict = "CRITICAL: Malware Source Download"

        attack_database.append({
            "time": datetime.datetime.now(MYT).strftime("%Y-%m-%d %H:%M:%S"),
            "ip": src_ip,
            "attacker_ip": src_ip,
            "target_ip": target_ip,
            "event": event_type,
            "details": details,
            "verdict": verdict,
            "reasoning": "Standard honeypot event. No AI vectoring required.",
            "advanced": "N/A"
        })

    save_db(attack_database)
    return jsonify({"status": "received"}), 200

@app.route('/api/analyze', methods=['GET'])
def analyze_threats():
    attack_database = get_db()
    total = len(attack_database)
    critical = sum(1 for attack in attack_database if "CRITICAL" in attack['verdict'])
    return jsonify({"total_attacks": total, "critical_threats": critical})

@app.route('/api/clear', methods=['POST'])
def clear_logs():
    save_db([])
    return jsonify({"status": "success", "message": "Database cleared successfully."})

@app.route('/api/chart-data', methods=['GET'])
def get_chart_data():
    attack_database = get_db()
    counts = {"CRITICAL": 0, "HIGH": 0, "WARNING": 0, "INFO": 0}
    attacker_ips = set()

    for attack in attack_database:
        verdict = attack.get('verdict', 'INFO: General Event')
        ip = attack.get('ip', attack.get('attacker_ip', 'Unknown IP'))
        if ip and ip != "Unknown IP":
            attacker_ips.add(ip)

        if "CRITICAL" in verdict: counts["CRITICAL"] += 1
        elif "HIGH" in verdict: counts["HIGH"] += 1
        elif "WARNING" in verdict: counts["WARNING"] += 1
        elif "INFO" in verdict: counts["INFO"] += 1

    total = sum(counts.values())
    insight_html = ""

    if total == 0:
        insight_html = "<span class='text-info fw-bold'>✅ System secure. Active listening on SSH honeypot. Awaiting network activity...</span>"
        border_class = "border-info"
    elif counts["CRITICAL"] > 0:
        ips_str = ", ".join(attacker_ips)
        insight_html = f"""
        <strong class='text-danger fs-5'>🚨 CRITICAL INCIDENT: MULTI-STAGE BREACH</strong>
        <ul class="mb-0 mt-2 text-light" style="font-size: 0.95em;">
            <li><strong>Adversary IP:</strong> {ips_str}</li>
        </ul>
        """
        border_class = "border-danger"
    else:
        insight_html = "<span class='text-success'>✅ Routine monitoring active.</span>"
        border_class = "border-success"

    return jsonify({"counts": counts, "insight_html": insight_html, "border_class": border_class})

@app.route('/api/export-report', methods=['GET'])
def export_report():
    attack_database = get_db()
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)

    pdf.add_page()

    def add_line(text, style='', size=10, align='L'):
        pdf.set_font("Courier", style=style, size=size)
        clean_text = text.encode('ascii', 'ignore').decode('ascii')
        pdf.multi_cell(0, 6, txt=clean_text, align=align)

    add_line("================================================================================", style='B')
    add_line(" [+] CYBERSENTINEL: AI-ENHANCED THREAT INTELLIGENCE & FORENSIC REPORT", style='B', size=11, align='C')
    add_line("================================================================================", style='B')
    add_line("")
    add_line(f"Report Generated: {datetime.datetime.now(MYT).strftime('%Y-%m-%d %H:%M:%S')} (Malaysia Time)")
    add_line("")

    if not attack_database:
        add_line("No events recorded in the current session.")
    else:
        for idx, attack in enumerate(attack_database, 1):
            ip_val = attack.get('ip', attack.get('attacker_ip', 'Unknown IP'))
            add_line(f"{idx}. [{attack['time']}] ATTACKER: {ip_val} -> TARGET: {attack.get('target_ip', 'N/A')}")
            add_line(f"    EVENT    : {attack['event']}")
            add_line(f"    PAYLOAD  : {attack['details'][:100]}")
            add_line(f"    VERDICT  : {attack['verdict']}")
            add_line("    ------------------------------------------------------------------")
            add_line("")

    filename = f"AI_Threat_Report_{datetime.datetime.now(MYT).strftime('%Y%m%d_%H%M%S')}.pdf"
    filepath = os.path.join(REPORT_FOLDER, filename)

    try:
        pdf.output(filepath)
        return send_file(filepath, as_attachment=True, download_name=filename)
    except Exception as e:
        print(f"PDF Generation Error: {e}")
        return jsonify({"error": "Failed to generate PDF"}), 500


if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)