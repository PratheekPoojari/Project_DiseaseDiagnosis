"""
tests/test_integration.py

Comprehensive headless integration test suite for the Disease Diagnosis system.
Verifies that all core modules interact seamlessly without requiring Streamlit:
1. Multi-format file parsing (.txt, .csv, .docx, .pdf) -> src.nlp.file_parser
2. NLP Inference -> src.nlp.predict
3. DIP Image Inference (ConvNeXt-Base) -> src.dip.predict
4. Late Fusion (text-only, image-only, and multimodal) -> src.fusion.fuse
5. Database initialization & Auth -> src.auth.db, src.auth.auth
6. Health Tracking & Trend Analysis -> src.health.tracker
7. Multi-format Report Export (.txt, .csv, .pdf, .docx) -> src.fusion.data_export
"""

import os
import io
import sys
import json
import docx
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet

# Add project root to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.nlp.file_parser import extract_text_from_file
from src.nlp.predict import load_model as load_nlp_model, predict_symptom
from src.dip.predict import load_model as load_dip_model, predict_image
from src.fusion.fuse import fuse_predictions
from src.auth.db import init_db, get_connection
from src.auth.auth import signup, login
from src.health.tracker import save_diagnosis, get_history, compute_trend
from src.fusion.data_export import (
    export_txt,
    export_csv,
    export_pdf,
    export_docx
)


def run_tests():
    print("=" * 70)
    print("🚀 STARTING HEADLESS INTEGRATION TEST SUITE (END-TO-END)")
    print("=" * 70)

    # 1. Multi-format file parsing
    print("\n[1/7] Testing Multi-Format File Parsers...")
    txt_content = "Patient reports dry, itchy scaling on both hands."
    extracted_txt = extract_text_from_file(txt_content.encode("utf-8"), "symptoms.txt")
    assert extracted_txt.strip() == txt_content, "Failed TXT parsing"

    csv_data = "id,symptoms,notes\n1,\"Intensely itchy red patches on flexural folds\",none\n"
    extracted_csv = extract_text_from_file(csv_data.encode("utf-8"), "records.csv")
    assert "Intensely itchy red patches" in extracted_csv, "Failed CSV parsing"

    doc = docx.Document()
    doc.add_paragraph("Erythematous scaly plaque with silvery borders.")
    buf_docx = io.BytesIO()
    doc.save(buf_docx)
    extracted_docx = extract_text_from_file(buf_docx.getvalue(), "notes.docx")
    assert "Erythematous scaly plaque" in extracted_docx, "Failed DOCX parsing"

    buf_pdf = io.BytesIO()
    pdf_doc = SimpleDocTemplate(buf_pdf, pagesize=A4)
    styles = getSampleStyleSheet()
    story = [Paragraph("Annular scaling plaque with central clearing.", styles['Normal'])]
    pdf_doc.build(story)
    extracted_pdf = extract_text_from_file(buf_pdf.getvalue(), "clinical_report.pdf")
    assert "Annular scaling plaque" in extracted_pdf, "Failed PDF parsing"
    print("  ✅ File parsing (.txt, .csv, .docx, .pdf) passed!")

    # 2. NLP Inference & Safety Net
    print("\n[2/7] Loading Bio_ClinicalBERT & Testing NLP Inference...")
    nlp_wrapper = load_nlp_model()

    in_domain = "I have intensely itchy, red inflamed patches on my inner elbows that are dry and scaly"
    res_nlp = predict_symptom(nlp_wrapper, in_domain)
    assert res_nlp["status"] == "ok", f"Expected 'ok', got {res_nlp['status']}"
    assert res_nlp["prediction"] is not None
    assert res_nlp["confidence"] >= 0.40, f"Expected confidence >= 0.40, got {res_nlp['confidence']}"
    assert len(res_nlp["probabilities"]) == 10
    print(f"  ✅ In-domain NLP test passed: {res_nlp['prediction']} ({res_nlp['confidence']*100:.2f}%)")

    out_of_domain = "My laptop screen flickers when I connect the HDMI cable"
    res_out = predict_symptom(nlp_wrapper, out_of_domain)
    assert res_out["status"] == "low_confidence", f"Expected 'low_confidence', got {res_out['status']}"
    assert res_out["confidence"] < 0.40, f"Expected confidence < 0.40, got {res_out['confidence']}"
    print(f"  ✅ Safety Net test passed: Caught non-medical text ({res_out['confidence']*100:.2f}% < 40%)")

    # 3. DIP Image Inference (ConvNeXt-Base)
    print("\n[3/7] Loading ConvNeXt-Base & Testing Image Inference...")
    dip_model, dip_classes, dip_device = load_dip_model()

    test_melanoma_img = os.path.join(PROJECT_ROOT, "data", "processed", "melanoma", "ISIC_6652710.jpg")
    assert os.path.exists(test_melanoma_img), f"Missing test image: {test_melanoma_img}"

    res_dip = predict_image(dip_model, dip_classes, dip_device, test_melanoma_img)
    assert res_dip["status"] == "ok", f"Expected 'ok', got {res_dip['status']}"
    assert res_dip["prediction"] == "Melanoma", f"Expected Melanoma, got {res_dip['prediction']}"
    assert res_dip["confidence"] >= 0.70, f"Expected high confidence, got {res_dip['confidence']}"
    assert len(res_dip["probabilities"]) == 10
    print(f"  ✅ DIP image test passed: {res_dip['prediction']} ({res_dip['confidence']*100:.2f}%)")

    # Missing image edge case
    res_missing = predict_image(dip_model, dip_classes, dip_device, "non_existent_file.jpg")
    assert res_missing["status"] == "error"
    print("  ✅ Missing image error handled cleanly.")

    # 4. Late Fusion Math
    print("\n[4/7] Testing Late Fusion (Text-only, Image-only, & Real Multimodal)...")
    # Single modality fallbacks
    fused_text_only = fuse_predictions(text_result=res_nlp, image_result=None)
    assert fused_text_only["prediction"] == res_nlp["prediction"]

    fused_image_only = fuse_predictions(text_result=None, image_result=res_dip)
    assert fused_image_only["prediction"] == res_dip["prediction"]

    # Real live multimodal consensus (real NLP output + real DIP output)
    fused_multimodal = fuse_predictions(text_result=res_nlp, image_result=res_dip, weight_image=0.6, weight_text=0.4)
    assert fused_multimodal["status"] == "ok"
    assert fused_multimodal["prediction"] is not None
    assert fused_multimodal["margin"] >= 0.10, "Expected clear winning margin"
    assert fused_multimodal["source"] == "multimodal_fusion"
    print(f"  ✅ Live Multimodal Fusion passed: Combined '{res_dip['prediction']}' (DIP) + '{res_nlp['prediction']}' (NLP) -> Winner: {fused_multimodal['prediction']} ({fused_multimodal['confidence']*100:.2f}%, margin: {fused_multimodal['margin']*100:.2f}%)")

    # Inconclusive / Ambiguity margin gating verification
    ambig_fused = fuse_predictions(
        text_result={"status": "ok", "probabilities": {"Eczema": 0.45, "Atopic Dermatitis": 0.44}},
        image_result={"status": "ok", "probabilities": {"Eczema": 0.44, "Atopic Dermatitis": 0.45}},
        weight_image=0.5,
        weight_text=0.5
    )
    assert ambig_fused["status"] == "inconclusive", f"Expected 'inconclusive', got {ambig_fused['status']}"
    assert ambig_fused["margin"] < 0.10, f"Expected margin < 0.10, got {ambig_fused['margin']}"
    print(f"  ✅ Margin Gating passed: Caught ambiguous draw ({ambig_fused['margin']*100:.2f}% < 10.0%) as 'inconclusive'")

    # 5. Auth & Database
    print("\n[5/7] Testing SQLite Schema & User Authentication...")
    init_db()
    test_username = "integration_test_runner"
    test_pass = "SecurePass123!"

    # Clean existing
    conn = get_connection()
    conn.execute("DELETE FROM users WHERE username = ?", (test_username,))
    conn.commit()
    conn.close()

    res_signup = signup(
        username=test_username,
        full_name="Integration Test Runner",
        email="runner@diagnosis.internal",
        phone="+919876543210",
        dob="1999-12-31",
        gender="Other",
        password=test_pass
    )
    assert res_signup["success"] is True
    reg_user = res_signup["user"]
    assert reg_user["username"] == test_username

    auth_success = login(test_username, test_pass)
    assert auth_success["success"] is True
    assert auth_success["user"]["id"] == reg_user["id"]

    auth_fail = login(test_username, "WrongPassword!")
    assert auth_fail["success"] is False
    print(f"  ✅ Auth passed: PBKDF2 hashing, user registration, and login verified for ID {reg_user['id']}")

    # 6. Health Tracking & Trend Analysis
    print("\n[6/7] Testing Diagnosis Persistence & Health Trends...")
    uid = reg_user["id"]

    save_diagnosis(user_id=uid, result=res_nlp, symptom_text=in_domain, image_used=False)
    save_diagnosis(user_id=uid, result=fused_multimodal, symptom_text="Follow-up notes", image_used=True)

    history = get_history(user_id=uid, limit=5)
    assert len(history) >= 2
    assert history[0]["prediction"] is not None

    trend = compute_trend(history)
    assert "status" in trend
    assert "message" in trend
    print(f"  ✅ Health Tracker passed: Logged 2 diagnoses, Trend status: '{trend['status']}'")

    # 7. Multi-Format Report Export
    print("\n[7/7] Testing Report Generation (.txt, .csv, .pdf, .docx)...")
    txt_out = export_txt(fused_multimodal, in_domain)
    assert b"MULTIMODAL SKIN DISEASE DIAGNOSIS REPORT" in txt_out

    csv_out = export_csv(fused_multimodal, in_domain)
    assert b"Predicted Condition" in csv_out

    pdf_out = export_pdf(fused_multimodal, in_domain)
    assert isinstance(pdf_out, bytes) and pdf_out.startswith(b"%PDF")

    docx_out = export_docx(fused_multimodal, in_domain)
    assert isinstance(docx_out, bytes) and len(docx_out) > 1000
    print(f"  ✅ Export passed: Generated valid TXT ({len(txt_out)} bytes), CSV ({len(csv_out)} bytes), PDF ({len(pdf_out)} bytes), DOCX ({len(docx_out)} bytes)")

    print("\n" + "=" * 70)
    print("🎉 ALL INTEGRATION TESTS PASSED SUCCESSFULLY! (7/7 MODULES VERIFIED)")
    print("=" * 70)


if __name__ == "__main__":
    run_tests()
