# Dataset Quality Metrics Module - Debug History

## Known Issues
None currently identified.

## Debugging Notes
- Ensure optional dependencies are handled gracefully
- Test with various dataset sizes for performance
- Validate metric calculations against known standards

## Resolved Issues
- [x] Fixed missing API client methods: 'APIClient' object has no attribute 'get_quality_history'
  - Added analyze_quality(), get_quality_results(), get_quality_history(), and list_datasets() methods to APIClient
- [x] Fixed dataset discovery showing "No datasets found" despite datasets existing in database
  - Updated get_available_datasets() to use proper API endpoints instead of hardcoded filesystem paths
  - Added fallback to legacy test_outputs directory for backward compatibility
- [x] Fixed dataset quality score calculation issue (always 0.0)
  - Identified root cause: double-serialization of dataset files (list of JSON strings instead of list of objects)
  - Updated `src/compileo/api/routes/quality.py` to correctly parse double-serialized datasets
  - Verified fix with reproduction script
- [x] Fixed AttributeError: 'NoneType' object has no attribute 'get' in quality history tab
  - Root cause: quality_jobs table missing from database initialization
  - Added quality_jobs table creation to initialize_database.py
  - Created table in existing database and added defensive null check in GUI
- [x] Fixed 400 Bad Request error when starting quality analysis
  - Root cause: project_id type mismatch (API expected int, database uses TEXT UUIDs)
  - Changed quality_jobs.project_id from INTEGER to TEXT in database schema
  - Removed int() conversion in API code and updated type hints

## Testing Notes
- Unit tests should cover edge cases for each metric
- Integration tests with datasetgen hooks
- Performance benchmarks for large datasets