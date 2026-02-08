# SpatialForge Security Audit Report

**Date**: 2026-02-08
**Auditor**: Claude Opus 4.6
**Framework**: OWASP API Security Top 10 (2023)
**Version**: SpatialForge v0.1.0

## Executive Summary

SpatialForge has **good baseline security** but requires **critical fixes** before production deployment. Overall security score: **7/10**.

### Critical Issues (Fix before production):
1. Default secret keys in config.py
2. No webhook endpoint rate limiting
3. Weak MinIO default credentials
4. No API key rotation mechanism

### High Priority (Fix within 1 week):
1. Add security.txt file
2. Implement stricter CSP
3. Add dependency vulnerability scanning
4. Enforce non-default secrets on startup

## OWASP API Security Top 10 Audit

### API1:2023 - Broken Object Level Authorization ✅ PASS

**Status**: No BOLA vulnerabilities found.

**Evidence**:
- All endpoints require API key authentication
- Redis-backed validation prevents unauthorized access
- No direct object access by ID without ownership verification
- Object store keys use UUIDs (not sequential IDs)

**Recommendation**: Continue good practices.

---

### API2:2023 - Broken Authentication ⚠️ WARNING

**Status**: Partially compliant with critical issues.

**Issues Found**:

1. **Default Secret Keys** (CRITICAL)
   - Location: `config.py` lines 14-15
   - Default `API_KEY_SECRET` = `"change-me-to-a-random-secret-key-at-least-32-chars"`
   - Default `ADMIN_API_KEY` = `"sf_admin_change_me"`
   - Impact: If deployed without env vars, attackers can forge valid API keys
   - **Fix**: Refuse to start if secrets are default values

2. **No API Key Rotation**
   - Once created, API keys never expire
   - No `expires_at` field in Redis data structure
   - **Fix**: Add key expiration mechanism

3. **No Rate Limiting on Webhook Endpoint**
   - Location: `api/v1/billing.py:202` (`/v1/billing/webhooks`)
   - No IP-based or signature-based rate limiting
   - **Fix**: Add rate limiter middleware specifically for webhooks

**Recommendations**:
```python
# In config.py, add startup validation:
def get_settings() -> Settings:
    s = Settings()
    if s.api_key_secret == _DEFAULT_SECRET and not s.demo_mode:
        raise RuntimeError("API_KEY_SECRET is using default value. Set via environment variable.")
    # ... existing validation
```

---

### API3:2023 - Broken Object Property Level Authorization ✅ PASS

**Status**: No property-level authorization issues.

**Evidence**:
- Pydantic models validate all input properties
- No mass assignment vulnerabilities
- Response models explicitly define exposed fields
- No raw database objects returned

---

### API4:2023 - Unrestricted Resource Consumption ✅ PASS (with minor notes)

**Status**: Good resource limits in place.

**Evidence**:
- File size limits: 20MB images, 100MB videos (enforced)
- Video duration limit: 120 seconds (enforced)
- Request timeout middleware: 120s on inference endpoints
- Rate limiting: Monthly quotas per API key (free=100, builder=5k, pro=50k)
- Redis rate limiter with sliding window

**Minor Issues**:
- No disk space checks before upload
- No GPU memory monitoring
- No concurrent request limits per API key

**Recommendations**:
- Add disk space pre-check in `storage/object_store.py`
- Add Prometheus alert for GPU memory >90%
- Consider per-key concurrent request limit (e.g., max 5 simultaneous)

---

### API5:2023 - Broken Function Level Authorization ✅ PASS

**Status**: Function-level authorization properly implemented.

**Evidence**:
- Admin endpoints (`/v1/admin/*`) check for `Plan.ADMIN` explicitly
- Billing endpoints validate plan tier (free/builder/pro cannot access enterprise features)
- No vertical privilege escalation vectors found
- Auth dependency injection pattern prevents bypass

**Code Review**:
```python
# api/v1/admin.py line 18-22
key_info = await get_auth(request)
if key_info.plan != Plan.ADMIN:
    raise HTTPException(status_code=403, detail="Admin access required")
```

---

### API6:2023 - Unrestricted Access to Sensitive Business Flows ⚠️ WARNING

**Status**: Partial protection, needs improvement.

**Issues Found**:

1. **No CAPTCHA on Expensive Endpoints**
   - `/v1/reconstruct`, `/v1/floorplan`, `/v1/segment-3d` (async GPU jobs)
   - Attackers with valid free-tier key can queue many jobs
   - Rate limiting is monthly, not concurrent
   - **Fix**: Add per-endpoint per-minute rate limit for expensive operations

2. **No Request Prioritization**
   - Free-tier and paid users share same Celery queue
   - No priority queue for paid customers
   - **Fix**: Implement separate queues: `gpu_heavy_paid` and `gpu_heavy_free`

**Recommendations**:
- Add per-endpoint rate limiter (e.g., max 10 reconstructions/hour for free tier)
- Implement queue prioritization by plan tier
- Consider adding CAPTCHA for anonymous/new API keys

---

### API7:2023 - Server Side Request Forgery (SSRF) ✅ PASS

**Status**: No SSRF vulnerabilities found.

**Evidence**:
- No user-controlled URLs in requests
- Webhook URLs validated (but not restricted to safe domains)
- No image URL fetching (only file uploads)
- MinIO endpoint is server-controlled, not user-supplied

**Minor Recommendation**:
- Add webhook URL whitelist for production (e.g., only allow HTTPS to known domains)

---

### API8:2023 - Security Misconfiguration ⚠️ WARNING

**Status**: Several misconfigurations found.

**Issues Found**:

1. **Weak MinIO Default Credentials** (HIGH)
   - Location: `config.py` lines 45-46
   - Default: `minioadmin/minioadmin`
   - **Fix**: Require strong credentials check on startup

2. **Missing security.txt** (MEDIUM)
   - No `/.well-known/security.txt` endpoint
   - Security researchers can't report vulnerabilities
   - **Fix**: Add static file with security contact

3. **Debug Mode Leakage Risk** (MEDIUM)
   - If `debug=True` in production, full stack traces exposed
   - No enforcement that `debug=False` for production origins
   - **Fix**: Add validation: `if "spatialforge-demo" in allowed_origins and debug: raise ValueError(...)`

4. **CORS Too Permissive for Demo Mode** (LOW)
   - Demo mode accepts requests from any origin
   - **Fix**: Document that demo mode should only be used in development

**Recommendations**:
```python
# Add to config validation
if s.minio_access_key == "minioadmin" and s.minio_secret_key == "minioadmin":
    logger.warning("MinIO using default credentials! Change MINIO_ACCESS_KEY and MINIO_SECRET_KEY")
```

---

### API9:2023 - Improper Inventory Management ✅ PASS

**Status**: Good API inventory and documentation.

**Evidence**:
- All endpoints documented in `site/docs.html`
- OpenAPI schema accessible at `/docs`
- Clear versioning (`/v1/*`)
- No deprecated endpoints without sunset notices
- Model licensing tracked in code (`model_manager.py`)

**Recommendations**:
- Add `Sunset` HTTP header when deprecating endpoints
- Consider API versioning strategy for v2

---

### API10:2023 - Unsafe Consumption of APIs ✅ PASS

**Status**: No unsafe API consumption found.

**Evidence**:
- Stripe webhook signature verification implemented
- No third-party API responses trusted without validation
- HuggingFace model downloads use HTTPS
- Redis connections require authentication

**Minor Issue**:
- Webhook signature is optional (`secret: str | None`)
- **Fix**: Make webhook signatures mandatory in production

---

## Additional Security Checks

### Input Validation ✅ PASS

- Pydantic models validate all inputs
- File type validation (magic bytes checked)
- Coordinate bounds checking
- NaN/Inf rejection
- SQL injection: N/A (no SQL database)
- Command injection: N/A (no shell commands from user input)

### Output Encoding ✅ PASS

- JSON responses properly encoded
- No XSS vectors (API-only, no HTML rendering)
- Content-Type headers set correctly

### Transport Security ✅ PASS

- HTTPS enforced via deployment (Fly.io)
- HSTS header enabled (SecurityHeadersMiddleware)
- TLS 1.2+ required by Fly.io

### Dependencies ⚠️ WARNING

**Issue**: No automated dependency vulnerability scanning

**Recommendations**:
- Add `pip-audit` to CI/CD pipeline
- Add Dependabot configuration for GitHub
- Run `safety check` periodically

**Current Known Vulnerabilities**: None identified (manual check needed)

---

## Security Headers Audit

**Current Headers** (from `SecurityHeadersMiddleware`):
- ✅ X-Content-Type-Options: nosniff
- ✅ X-Frame-Options: DENY
- ✅ Referrer-Policy: strict-origin-when-cross-origin
- ✅ Permissions-Policy: (restrictive)
- ✅ X-API-Version: 0.1.0
- ✅ Strict-Transport-Security (production only)
- ⚠️ Content-Security-Policy: Permissive for docs

**CSP Issues**:
- Current CSP allows `'unsafe-inline'` for styles (docs page needs it)
- No CSP for API endpoints (acceptable for JSON API)
- **Recommendation**: Split CSP policy for `/docs` vs `/v1/*`

---

## Threat Model

### Identified Threats:

1. **API Key Theft** (Medium)
   - Mitigation: API keys transmitted via HTTPS only
   - Gap: No key rotation, no revocation UI

2. **DoS via Expensive Operations** (Medium)
   - Mitigation: Rate limiting, timeouts
   - Gap: No per-endpoint rate limits

3. **Data Exfiltration** (Low)
   - Mitigation: Presigned URLs with TTL (24 hours)
   - No PII stored (only API keys)

4. **Supply Chain Attack** (Medium)
   - Mitigation: Apache 2.0 models only (license enforcement)
   - Gap: No dependency vulnerability scanning

5. **GPU Resource Exhaustion** (High)
   - Mitigation: Celery queues, max retries
   - Gap: No GPU memory limits per job

---

## Compliance Considerations

### GDPR Compliance:
- ✅ No PII collected (API keys are not PII)
- ✅ Data retention: 24-hour TTL on results
- ⚠️ No data processing agreement template
- ⚠️ No GDPR-compliant cookie policy

### CCPA Compliance:
- ✅ No personal information sold
- N/A (business API, not consumer-facing)

### SOC 2 Readiness:
- ⚠️ Missing: Audit logs
- ⚠️ Missing: Access logs retention policy
- ⚠️ Missing: Incident response plan

---

## Recommended Fixes (Prioritized)

### Critical (Fix Today):
1. Enforce non-default secrets on startup
2. Add webhook endpoint rate limiting

### High Priority (Fix This Week):
3. Implement API key rotation mechanism
4. Add security.txt file
5. Strengthen MinIO credentials validation
6. Add per-endpoint rate limits for GPU-heavy operations

### Medium Priority (Fix This Month):
7. Add dependency vulnerability scanning (pip-audit)
8. Implement queue prioritization by plan tier
9. Add Prometheus alerts for GPU memory
10. Create incident response runbook

### Low Priority (Nice to Have):
11. Add webhook URL whitelist
12. Implement API key revocation UI
13. Add audit logging for admin actions
14. Create data processing agreement template

---

## Security Scoring

| Category | Score | Rationale |
|----------|-------|-----------|
| Authentication | 6/10 | Default keys, no rotation |
| Authorization | 9/10 | Strong RBAC, proper checks |
| Input Validation | 10/10 | Comprehensive Pydantic validation |
| Rate Limiting | 7/10 | Good global limits, missing per-endpoint |
| Secrets Management | 5/10 | Weak defaults, no rotation |
| Dependency Security | 6/10 | No automated scanning |
| Monitoring | 7/10 | Prometheus metrics, missing alerts |
| Incident Response | 4/10 | No documented plan |
| **Overall** | **7/10** | **Good foundation, needs hardening** |

---

## Action Plan

### Immediate (before production launch):
```bash
# 1. Add startup secret validation
# 2. Add webhook rate limiter
# 3. Add security.txt
# 4. Run dependency scan
pip install pip-audit safety
pip-audit
safety check
```

### Week 1:
- Implement API key rotation
- Add per-endpoint rate limits
- Configure Prometheus alerts

### Month 1:
- Implement queue prioritization
- Add audit logging
- Create incident response plan

---

## Conclusion

SpatialForge demonstrates **good security awareness** with proper authentication, authorization, input validation, and security headers. However, **critical production readiness issues** must be addressed:

1. **Remove all default secrets** and enforce strong credentials
2. **Add comprehensive rate limiting** for expensive endpoints
3. **Implement security monitoring** with Prometheus alerts

After addressing these issues, SpatialForge will be **production-ready** with a security score of **8.5/10**.

---

**Report Generated**: 2026-02-08
**Next Audit**: Recommended after major feature additions or every 6 months
**Contact**: security@spatialforge.example.com (add this!)
