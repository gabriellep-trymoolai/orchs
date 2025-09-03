"""
Firewall and Security Scanning API Router
=========================================

Provides endpoints for security scanning including PII detection, 
secrets detection, and toxicity detection.
"""

import re
import time
import math
import hashlib
from datetime import datetime
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field


router = APIRouter(prefix="/api/v1/firewall", tags=["firewall", "security"])


class ScanRequest(BaseModel):
    """Request model for security scanning"""
    content: str = Field(..., description="Content to scan for security issues")
    user_id: Optional[str] = Field(None, description="User ID for tracking")
    scan_id: Optional[str] = Field(None, description="Custom scan ID")


class ScanResult(BaseModel):
    """Base result model for security scans"""
    scan_id: str
    content_length: int
    scan_type: str
    risk_level: str  # "low", "medium", "high", "critical"
    issues_found: int
    scan_time_ms: int
    timestamp: datetime


class PIIScanResult(ScanResult):
    """Result model for PII scanning"""
    detected_pii: List[Dict[str, Any]]
    pii_types: List[str]
    confidence_score: float


class SecretsScanResult(ScanResult):
    """Result model for secrets scanning"""
    detected_secrets: List[Dict[str, Any]]
    secret_types: List[str]
    entropy_analysis: Dict[str, float]


class ToxicityScanResult(ScanResult):
    """Result model for toxicity scanning"""
    toxicity_score: float
    detected_categories: List[str]
    flagged_phrases: List[Dict[str, Any]]


# PII Detection patterns
PII_PATTERNS = {
    "email": {
        "pattern": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        "description": "Email address"
    },
    "ssn": {
        "pattern": r'\b\d{3}-?\d{2}-?\d{4}\b',
        "description": "Social Security Number"
    },
    "phone": {
        "pattern": r'\b(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b',
        "description": "Phone number"
    },
    "credit_card": {
        "pattern": r'\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13}|3[0-9]{13}|6(?:011|5[0-9]{2})[0-9]{12})\b',
        "description": "Credit card number"
    },
    "ip_address": {
        "pattern": r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b',
        "description": "IP address"
    },
    "driver_license": {
        "pattern": r'\b[A-Z]{1,2}[0-9]{6,8}\b',
        "description": "Driver's license number"
    }
}

# Secrets detection patterns
SECRETS_PATTERNS = {
    "api_key": {
        "pattern": r'\b[Aa][Pp][Ii][-_]?[Kk][Ee][Yy]\s*[:=]\s*[\'"]?([A-Za-z0-9_-]{20,})[\'"]?',
        "description": "API key"
    },
    "aws_access_key": {
        "pattern": r'\b(AKIA[0-9A-Z]{16})\b',
        "description": "AWS access key"
    },
    "github_token": {
        "pattern": r'\bghp_[A-Za-z0-9]{36}\b',
        "description": "GitHub personal access token"
    },
    "jwt_token": {
        "pattern": r'\beyJ[A-Za-z0-9_-]*\.[A-Za-z0-9_-]*\.[A-Za-z0-9_-]*\b',
        "description": "JWT token"
    },
    "password": {
        "pattern": r'\b[Pp][Aa][Ss][Ss][Ww][Oo][Rr][Dd]\s*[:=]\s*[\'"]?([A-Za-z0-9@#$%^&*!]{8,})[\'"]?',
        "description": "Password"
    },
    "private_key": {
        "pattern": r'-----BEGIN [A-Z ]+PRIVATE KEY-----',
        "description": "Private key"
    }
}

# Toxicity detection keywords
TOXICITY_CATEGORIES = {
    "hate_speech": [
        "hate", "racist", "nazi", "fascist", "bigot", "supremacist"
    ],
    "harassment": [
        "harassment", "bully", "threaten", "intimidate", "stalk"
    ],
    "violence": [
        "kill", "murder", "assault", "attack", "violence", "harm", "hurt"
    ],
    "profanity": [
        "damn", "hell", "shit", "fuck", "bitch", "asshole"  # Basic examples
    ],
    "discrimination": [
        "discriminate", "prejudice", "bias", "stereotype"
    ]
}


def calculate_entropy(text: str) -> float:
    """Calculate Shannon entropy of text to detect random strings (potential secrets)"""
    if not text:
        return 0.0
    
    # Count character frequencies
    char_counts = {}
    for char in text:
        char_counts[char] = char_counts.get(char, 0) + 1
    
    # Calculate entropy
    entropy = 0.0
    text_length = len(text)
    for count in char_counts.values():
        probability = count / text_length
        entropy -= probability * math.log2(probability)
    
    return entropy


def detect_pii(content: str) -> List[Dict[str, Any]]:
    """Detect PII in content using regex patterns"""
    detected_pii = []
    
    for pii_type, pattern_info in PII_PATTERNS.items():
        pattern = pattern_info["pattern"]
        matches = re.finditer(pattern, content)
        
        for match in matches:
            detected_pii.append({
                "type": pii_type,
                "description": pattern_info["description"],
                "value": match.group(),
                "start_pos": match.start(),
                "end_pos": match.end(),
                "confidence": 0.8  # Static confidence for regex matches
            })
    
    return detected_pii


def detect_secrets(content: str) -> tuple[List[Dict[str, Any]], Dict[str, float]]:
    """Detect secrets and analyze entropy"""
    detected_secrets = []
    
    for secret_type, pattern_info in SECRETS_PATTERNS.items():
        pattern = pattern_info["pattern"]
        matches = re.finditer(pattern, content, re.IGNORECASE)
        
        for match in matches:
            secret_value = match.group(1) if match.groups() else match.group()
            entropy = calculate_entropy(secret_value)
            
            detected_secrets.append({
                "type": secret_type,
                "description": pattern_info["description"],
                "value": secret_value[:10] + "..." if len(secret_value) > 10 else secret_value,
                "start_pos": match.start(),
                "end_pos": match.end(),
                "entropy": entropy,
                "confidence": 0.9 if entropy > 4.0 else 0.7
            })
    
    # Calculate overall entropy analysis
    entropy_analysis = {
        "overall_entropy": calculate_entropy(content),
        "high_entropy_segments": 0,
        "average_word_entropy": 0.0
    }
    
    # Analyze words for high entropy
    words = re.findall(r'\b\w{8,}\b', content)  # Words with 8+ characters
    if words:
        word_entropies = [calculate_entropy(word) for word in words]
        entropy_analysis["average_word_entropy"] = sum(word_entropies) / len(word_entropies)
        entropy_analysis["high_entropy_segments"] = sum(1 for e in word_entropies if e > 4.0)
    
    return detected_secrets, entropy_analysis


def detect_toxicity(content: str) -> tuple[float, List[str], List[Dict[str, Any]]]:
    """Detect toxic content using keyword matching"""
    content_lower = content.lower()
    detected_categories = []
    flagged_phrases = []
    total_toxicity_score = 0.0
    
    for category, keywords in TOXICITY_CATEGORIES.items():
        category_score = 0.0
        category_matches = []
        
        for keyword in keywords:
            if keyword in content_lower:
                # Find all occurrences
                start = 0
                while True:
                    pos = content_lower.find(keyword, start)
                    if pos == -1:
                        break
                    
                    category_matches.append({
                        "phrase": keyword,
                        "position": pos,
                        "context": content[max(0, pos-20):pos+len(keyword)+20],
                        "severity": "medium"  # Could be enhanced with severity levels
                    })
                    category_score += 0.3  # Each keyword adds to category score
                    start = pos + 1
        
        if category_matches:
            detected_categories.append(category)
            flagged_phrases.extend(category_matches)
            total_toxicity_score += min(category_score, 1.0)  # Cap per category at 1.0
    
    # Normalize toxicity score to 0-1 range
    toxicity_score = min(total_toxicity_score / len(TOXICITY_CATEGORIES), 1.0)
    
    return toxicity_score, detected_categories, flagged_phrases


def determine_risk_level(issues_count: int, max_severity: float = 1.0) -> str:
    """Determine risk level based on issues found and severity"""
    if issues_count == 0:
        return "low"
    elif issues_count <= 2 and max_severity < 0.5:
        return "medium"
    elif issues_count <= 5 and max_severity < 0.8:
        return "high"
    else:
        return "critical"


@router.post("/scan/pii", response_model=PIIScanResult)
async def scan_pii(request: ScanRequest):
    """
    Scan content for Personally Identifiable Information (PII).
    
    Detects:
    - Email addresses
    - Social Security Numbers
    - Phone numbers
    - Credit card numbers
    - IP addresses
    - Driver's license numbers
    
    Args:
        request: Content to scan for PII
        
    Returns:
        PIIScanResult: Detailed PII scan results
    """
    start_time = time.time()
    scan_id = request.scan_id or f"pii_scan_{int(time.time())}"
    
    try:
        # Perform PII detection
        detected_pii = detect_pii(request.content)
        
        # Extract unique PII types
        pii_types = list(set(item["type"] for item in detected_pii))
        
        # Calculate confidence score
        confidence_score = 0.0
        if detected_pii:
            confidence_score = sum(item["confidence"] for item in detected_pii) / len(detected_pii)
        
        # Determine risk level
        risk_level = determine_risk_level(len(detected_pii))
        
        scan_time_ms = int((time.time() - start_time) * 1000)
        
        return PIIScanResult(
            scan_id=scan_id,
            content_length=len(request.content),
            scan_type="pii",
            risk_level=risk_level,
            issues_found=len(detected_pii),
            scan_time_ms=scan_time_ms,
            timestamp=datetime.now(),
            detected_pii=detected_pii,
            pii_types=pii_types,
            confidence_score=confidence_score
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PII scan failed: {str(e)}")


@router.post("/scan/secrets", response_model=SecretsScanResult)
async def scan_secrets(request: ScanRequest):
    """
    Scan content for secrets and sensitive information.
    
    Detects:
    - API keys
    - AWS access keys
    - GitHub tokens
    - JWT tokens
    - Passwords
    - Private keys
    
    Also performs entropy analysis to detect potential secrets.
    
    Args:
        request: Content to scan for secrets
        
    Returns:
        SecretsScanResult: Detailed secrets scan results
    """
    start_time = time.time()
    scan_id = request.scan_id or f"secrets_scan_{int(time.time())}"
    
    try:
        # Perform secrets detection
        detected_secrets, entropy_analysis = detect_secrets(request.content)
        
        # Extract unique secret types
        secret_types = list(set(item["type"] for item in detected_secrets))
        
        # Determine risk level based on secrets found and entropy
        max_entropy = max([item["entropy"] for item in detected_secrets] + [0])
        risk_level = determine_risk_level(len(detected_secrets), max_entropy / 6.0)  # Normalize entropy
        
        scan_time_ms = int((time.time() - start_time) * 1000)
        
        return SecretsScanResult(
            scan_id=scan_id,
            content_length=len(request.content),
            scan_type="secrets",
            risk_level=risk_level,
            issues_found=len(detected_secrets),
            scan_time_ms=scan_time_ms,
            timestamp=datetime.now(),
            detected_secrets=detected_secrets,
            secret_types=secret_types,
            entropy_analysis=entropy_analysis
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Secrets scan failed: {str(e)}")


@router.post("/scan/toxicity", response_model=ToxicityScanResult)
async def scan_toxicity(request: ScanRequest):
    """
    Scan content for toxic, harmful, or inappropriate content.
    
    Detects:
    - Hate speech
    - Harassment
    - Violence
    - Profanity
    - Discrimination
    
    Args:
        request: Content to scan for toxicity
        
    Returns:
        ToxicityScanResult: Detailed toxicity scan results
    """
    start_time = time.time()
    scan_id = request.scan_id or f"toxicity_scan_{int(time.time())}"
    
    try:
        # Perform toxicity detection
        toxicity_score, detected_categories, flagged_phrases = detect_toxicity(request.content)
        
        # Determine risk level based on toxicity score
        if toxicity_score == 0:
            risk_level = "low"
        elif toxicity_score < 0.3:
            risk_level = "medium"
        elif toxicity_score < 0.7:
            risk_level = "high"
        else:
            risk_level = "critical"
        
        scan_time_ms = int((time.time() - start_time) * 1000)
        
        return ToxicityScanResult(
            scan_id=scan_id,
            content_length=len(request.content),
            scan_type="toxicity",
            risk_level=risk_level,
            issues_found=len(flagged_phrases),
            scan_time_ms=scan_time_ms,
            timestamp=datetime.now(),
            toxicity_score=toxicity_score,
            detected_categories=detected_categories,
            flagged_phrases=flagged_phrases
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Toxicity scan failed: {str(e)}")


@router.post("/scan/comprehensive")
async def comprehensive_scan(request: ScanRequest):
    """
    Perform a comprehensive security scan including PII, secrets, and toxicity detection.
    
    Args:
        request: Content to scan
        
    Returns:
        Combined results from all scan types
    """
    start_time = time.time()
    scan_id = request.scan_id or f"comprehensive_scan_{int(time.time())}"
    
    try:
        # Perform all scans
        pii_results = await scan_pii(request)
        secrets_results = await scan_secrets(request)
        toxicity_results = await scan_toxicity(request)
        
        # Calculate overall risk
        risk_levels = ["low", "medium", "high", "critical"]
        max_risk = max([
            risk_levels.index(pii_results.risk_level),
            risk_levels.index(secrets_results.risk_level),
            risk_levels.index(toxicity_results.risk_level)
        ])
        
        overall_risk = risk_levels[max_risk]
        total_issues = pii_results.issues_found + secrets_results.issues_found + toxicity_results.issues_found
        
        scan_time_ms = int((time.time() - start_time) * 1000)
        
        return {
            "scan_id": scan_id,
            "scan_type": "comprehensive",
            "content_length": len(request.content),
            "overall_risk_level": overall_risk,
            "total_issues_found": total_issues,
            "scan_time_ms": scan_time_ms,
            "timestamp": datetime.now(),
            
            # Individual scan results
            "pii_scan": pii_results,
            "secrets_scan": secrets_results,
            "toxicity_scan": toxicity_results,
            
            # Summary
            "summary": {
                "pii_detected": pii_results.issues_found > 0,
                "secrets_detected": secrets_results.issues_found > 0,
                "toxicity_detected": toxicity_results.issues_found > 0,
                "highest_risk_category": overall_risk
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Comprehensive scan failed: {str(e)}")


@router.get("/health")
async def firewall_health_check():
    """
    Check firewall service health and capability.
    
    Returns:
        Health status and available scan types
    """
    return {
        "status": "healthy",
        "available_scans": ["pii", "secrets", "toxicity", "comprehensive"],
        "scan_patterns": {
            "pii_types": list(PII_PATTERNS.keys()),
            "secret_types": list(SECRETS_PATTERNS.keys()),
            "toxicity_categories": list(TOXICITY_CATEGORIES.keys())
        },
        "timestamp": datetime.now()
    }