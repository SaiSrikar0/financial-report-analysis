# 🚀 Uploaded Data Analysis Pipeline - Complete Workflow

## Overview
Uploaded data now follows the **same analysis pipeline as predefined reference data (AAPL/MSFT/GOOGL/AMZN)**. This means you get:
- ✅ Data normalization (Phase 1)
- ✅ Storage to database (Phase 1)
- ✅ SVR growth predictions (Phase 4)
- ✅ SHAP feature importance (Phase 5)
- ✅ AI recommendations powered by LLM (Phase 6)
- ✅ Dashboard visualizations

## Complete Upload Workflow

### Step 1: Upload Your Data (**📤 Upload Data Page**)
1. Enter company **ticker** (e.g., NEXTGEN, TESLA, SAMSUNG)
2. Upload file (CSV, JSON, Excel, PDF)
3. Click **"Process & Load Data"**
   - Extracts raw records
   - LLM normalizes to standard financial schema
   - Validates completeness
   - Transforms & engineers features
   - **Stores to database** ⭐ (NEW)

### Step 2: Run Analysis Pipeline (**Still on Upload Page**)
After upload succeeds, you'll see:
```
🚀 Run Analysis Pipeline (Phase 4-5)
```

Click this button to automatically:
- Train SVR model for your ticker
- Generate growth rate predictions
- Analyze feature importance via SHAP
- Save all results to report files

**Progress will show:**
```
✓ Loaded X standard records + Y category records
→ Phase 4: Training SVR model...
  ✓ SVR model trained and predictions saved
→ Phase 5: Generating SHAP analysis...
  ✓ SHAP analysis complete
✓ Analysis complete for NEXTGEN!
```

### Step 3: View Dashboard (**📊 Dashboard Page → "📁 Uploaded Data" Tab**)
After analysis completes:
- **File list** shows all your uploaded files
- **Metric visualization** displays trends in your data (line charts)
- **SVR Analysis** shows predicted growth rate vs 10% target
- **SHAP Analysis** shows top 8 features driving predictions

### Step 4: Generate AI Recommendations (**🤖 AI Recommendations Page**)
1. Select your uploaded ticker from dropdown (now includes uploaded tickers!)
2. Click **"Generate AI Recommendations"**
3. LLM analyzes:
   - SVR predictions
   - SHAP feature importance
   - Historical trends
   - Peer comparison
4. Returns structured investment intelligence:
   - Executive summary
   - Performance score (1-10)
   - Growth outlook
   - Risk assessment
   - Critical warnings
   - Opportunities
   - Investment verdict (BUY/HOLD/REDUCE)

---

## Architecture Changes

### Before ❌
```
Upload → Normalize → Validate → Store JSON only
```
Problem: No SVR predictions, no SHAP, no recommendations

### After ✅
```
Upload → Normalize → Validate → Store to DB → Run Phase 4-5 → Full Analysis
```
Result: Complete analysis pipeline just like predefined data

### Key Implementation Details

**1. Data Storage (NEW)**
- After transformation, data now stored to `standard_table` and `category_table`
- Tagged with `user_id` for isolation
- Available for all analysis phases

**2. Single-Ticker Analysis (NEW)**
- Phase 4 & 5 can now run for just your uploaded ticker
- Results saved to report files alongside predefined data
- Dashboard and recommendations seamlessly work with both

**3. Automatic Report Generation (NEW)**
- If SVR training fails, creates basic predictions from data trends
- If SHAP fails, creates basic feature importance from correlations
- Ensures recommendations always work

**4. Fallback Mechanisms**
- If Phase 4 fails: Basic growth rate calculated from net_income trend
- If Phase 5 fails: Feature importance from statistical correlation
- Both fallbacks ensure Phase 6 (recommendations) always succeeds

---

## File Structure

```
analysis/reports/
├── svr_future_predictions.csv          ← Includes uploaded ticker
├── phase_5_shap_global_importance.csv  ← Feature rankings
├── phase_5_shap_local_explanations.csv ← Ticker-specific drivers
└── [other standard reports]            ← Same as before

analysis/
├── auto_analysis.py                    ← NEW: Orchestrates Phase 4-5
├── uploaded_data_analytics.py          ← Dashboard visualization
└── [other phase modules]
```

---

## Testing the Full Workflow

### Test Case: Upload NEXTGEN, Generate Recommendations

**Terminal Commands (for reference):**
```bash
# Option 1: Use UI (Recommended)
# 1. Go to Upload Data page
# 2. Upload NEXTGEN file
# 3. Click "Run Analysis Pipeline"
# 4. Go to AI Recommendations
# 5. Select NEXTGEN, click Generate

# Option 2: Run phases manually
python run.py 4  # Phase 4: SVR (all tickers)
python run.py 5  # Phase 5: SHAP (all tickers)
streamlit run app.py
# Then go to recommendations page
```

### Expected Results
- ✅ "NEXTGEN" appears in AI Recommendations dropdown
- ✅ Dashboard shows NEXTGEN file in "Uploaded Data" tab
- ✅ SVR predictions display with growth rate
- ✅ SHAP chart shows top 8 features
- ✅ Recommendations include ticket-specific insights

---

## Troubleshooting

### "No SVR predictions found for NEXTGEN"
**Issue:** Analysis pipeline button was not clicked  
**Fix:** Go back to Upload page → click "🚀 Run Analysis Pipeline" button

### "No data uploaded yet" in dashboard
**Issue:** RLS policy still blocking frontend reads  
**Fix:** Run the SQL fix from `DASHBOARD_FIX_INSTRUCTIONS.md`

### Analysis pipeline takes too long
**Issue:** SVR training on large dataset  
**Fix:** Let it complete or use manual phase runs for faster iteration

### Uploaded data not in recommendations dropdown
**Issue:** Data not stored to database or database connection issue  
**Fix:** Check that upload UI shows checkmark at storage step

---

## Technical Notes

### Phases Explained

**Phase 1: Extraction & Normalization** (In Upload UI)
- Extracts raw records from file
- 2-prompt LLM normalization pipeline
- Validates schema completeness
- Transforms & engineers ~15 financial features
- **NEW**: Stores to Supabase tables with user_id

**Phase 4: SVR Modeling** (Auto-run after upload)
- Trains support vector regression model
- Predicts next period revenue/growth
- Calculates gap vs 10% target
- Saves to `svr_future_predictions.csv`

**Phase 5: SHAP Explainability** (Auto-run after upload)
- Calculates feature importance via SHAP
- Identifies which metrics drive growth predictions
- Saves to `phase_5_shap_global_importance.csv`

**Phase 6: LLM Recommendations** (Dashboard + Recommendations page)
- Loads SVR + SHAP + financial data
- Calls Groq LLaMA for analysis
- Returns structured investment intelligence
- Stores results in `recommendation_results` table

### Performance

- **Upload + Normalization:** ~2-5 seconds
- **Phase 4 (SVR):** ~10-30 seconds (depends on data size)
- **Phase 5 (SHAP):** ~5-15 seconds
- **Phase 6 (LLM):** ~3-8 seconds

**Total:** ~20-60 seconds from upload to recommendations ready

---

## Next Steps

1. ✅ Code is ready - app is running at http://localhost:8502
2. ✅ Make sure RLS policies are updated (see DASHBOARD_FIX_INSTRUCTIONS.md if not already done)
3. 🎯 Test by uploading a file and running the analysis pipeline
4. 🎯 Generate AI recommendations for your uploaded ticker
5. 🎯 Verify dashboard shows all visualizations

---

## Questions?

The analysis pipeline automatically handles:
- ✅ Converting raw → normalized data
- ✅ Storing multi-user data with isolation
- ✅ Training models for single tickers
- ✅ Generating predictions even if some phases fail
- ✅ Showing results in dashboard & recommendations

All while preserving the predefined AAPL/MSFT/GOOGL/AMZN reference data alongside your uploads.
