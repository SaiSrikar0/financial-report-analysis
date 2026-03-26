# 📊 SVR Training Data Retrieval Architecture

## Overview
The SVR training pipeline now has a robust, multi-source data retrieval system that intelligently loads uploaded data and validates it contains all required financial parameters.

---

## Data Flow Architecture

### Source 1: Upload Flow (Immediate)
```
User Upload 
  → Extract raw records
  → LLM normalize to standard schema
  → Store to uploaded_files (JSON)
  → Store to standard_table (normalized)
  ↓
Training Pipeline gets:
  ✓ standard_records (already normalized)
  ✓ category_records
  → Validates immediately
  → Trains successfully
```

### Source 2: Dashboard/Recommendations (From Database)
```
User clicks "Train SVR" on existing ticker
  ↓
System retrieves from database:
  1. Try standard_table first (already normalized)
  2. If empty, try uploaded_files table (raw JSON)
  ↓
Validates all required fields present
  ↓
Trains model
```

---

## Required Financial Parameters

### Critical Fields (Must Have)
```
- date              (temporal data point)
- ticker            (company identifier)
- revenue           (top line)
- net_income        (bottom line)
- operating_income  (core profitability)
- total_assets      (balance sheet)
- total_liabilities (balance sheet)
- operating_cashflow(actual cash generation)
```

### Optional Fields (Enhance Model)
```
- profit_margin
- operating_margin
- revenue_growth
- net_income_growth
- asset_efficiency
- debt_to_asset
```

---

## New Module: `analysis/data_retrieval_svr.py`

### Key Functions

#### 1. `retrieve_uploaded_data_by_ticker(ticker, user_id, supabase_client)`
**Purpose:** Intelligently load data from any source
**Logic:**
```
Priority 1: Load from standard_table (already normalized)
  ↓ If empty/error
Priority 2: Load from uploaded_files table (raw JSON)
  ↓ If both fail
Return: (None, "error_message")
```

**Returns:** `(DataFrame, source)` or `(None, error_message)`

---

#### 2. `validate_and_prepare_svr_data(df, ticker, min_records=3)`
**Purpose:** Ensure data quality for model training
**Validations:**
- ✓ Minimum record count (default 3)
- ✓ Has all required financial fields
- ✓ No multiple tickers mixed in
- ✓ Correct data types (numeric, datetime)
- ✓ No NaN values in required fields

**Returns:** `(cleaned_df, validation_messages)`

**Example Output:**
```
✓ Prepared 45 clean records for training
ℹ️ Removed 2 rows with missing 'operating_income'
⚠️ Only 45 records (need ≥3) — adequate but edge case
```

---

#### 3. `load_and_validate_training_data(ticker, user_id, supabase_client, standard_records=None, category_records=None)`
**Purpose:** Complete retrieval + validation pipeline
**Flow:**
1. If records provided (upload flow) → use directly
2. Otherwise retrieve from database
3. Validate all fields present
4. Check minimum data quantity
5. Return clean DataFrame + detailed reports

**Returns:** `(prepared_df, validation_messages)`

---

## Integration with SVR Training

### Old Flow (Broken) ❌
```python
# Would fail silently on bad data
run_uploaded_analysis_pipeline(
    ticker,
    standard_records,      # Could be incomplete
    category_records,      # Could be empty
    user_id
)
```

### New Flow (Robust) ✅
```python
# Validates everything first
run_uploaded_analysis_pipeline(
    ticker,
    user_id,
    supabase_client,       # Client with user session
    standard_records=None,  # Optional - will load if not provided
    category_records=None   # Optional
)

# Inside: data_retrieval_svr validates before training
# Shows detailed progress: loading → validating → training
```

---

## Error Handling Scenarios

### Scenario 1: Upload Flow
```
✓ Using provided uploaded records
→ Validating data completeness...
✓ Prepared 35 clean records for training
→ Phase 4: Training SVR model...
✓ SVR model trained
```

### Scenario 2: Dashboard - Existing Data
```
→ Retrieving from database...
✓ Loaded 35 records from standard_table
→ Validating data completeness...
✓ Prepared 35 clean records for training
→ Phase 4: Training SVR model...
✓ SVR model trained
```

### Scenario 3: Dashboard - Missing Fields
```
→ Retrieving from database...
✓ Loaded 35 records from uploaded_files
→ Validating data completeness...
⚠️ Missing fields: debt_to_asset, asset_efficiency
✅ Data Ready: 35 records with all required fields
→ Phase 4: Training SVR model...
✓ SVR model trained
```

### Scenario 4: Insufficient Data
```
→ Retrieving from database...
✓ Loaded 2 records from standard_table
→ Validating data completeness...
⚠️ Only 2 records (need ≥3)
❌ CRITICAL: Only 2 valid records after cleaning
❌ Data validation failed - cannot proceed with training
```

---

## Data Source Priority

### For Training, System Tries (In Order):
1. **standard_table** (if has ticker + user_id)
   - Already normalized by LLM
   - Ready to train immediately
   - Best performance
   
2. **uploaded_files** (if standard_table empty)
   - Raw JSON data from upload
   - Will be used as-is
   - Still requires validation

3. **Provided records** (in upload flow)
   - Direct from normalization pipeline
   - Already validated in upload flow
   - Fastest path

---

## Validation Report Example

```
→ Loading and validating data...
→ Retrieving from database...
✓ Loaded 45 records from standard_table for NEXTGEN

→ Validating data completeness...
ℹ️ Removed 1 row with missing 'revenue'
ℹ️ Removed 2 rows with missing 'net_income'
✓ Prepared 42 clean records for training

✅ Data Ready: 42 records with all required fields
```

---

## Debugging Tips

**If "No data found" error:**
1. Check data was uploaded to dashboard
2. Verify data stored to `standard_table` (check DB)
3. Check ticker spelling matches exactly
4. Re-upload file if table is empty

**If "Missing fields" warning:**
1. Your upload file is missing financial metrics
2. Adjust your source data to include all fields
3. Or data will still train with whatever's available

**If SVR training still fails:**
1. System shows detailed validation messages
2. Check that at least 3 records after cleaning
3. Check all $ amounts are numeric (not text)
4. Check dates are valid YYYY-MM-DD format

---

## Technical Improvements

✅ **Before:** Silent failures when data missing  
✅ **Now:** Detailed validation with clear messaging

✅ **Before:** Required exact source type known  
✅ **Now:** Intelligently tries multiple sources

✅ **Before:** No way to debug data issues  
✅ **Now:** Step-by-step validation report

✅ **Before:** Upload and dashboard had different logic  
✅ **Now:** Unified retrieval + validation logic

---

## Next Steps

When user clicks "Train SVR":
1. System retrieves data (shows source)
2. System validates fields + records (shows issues)
3. System trains model OR shows why it can't
4. User sees clear progress + any warnings
