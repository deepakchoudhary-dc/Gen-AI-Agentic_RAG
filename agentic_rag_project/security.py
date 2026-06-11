"""
Security, Privacy, & Enterprise Guardrails Module

Implements:
  - Data Anonymization / PII Masking
  - Prompt Injection Defense
  - Role-Based Access Control (RBAC)
  - Air-Gapped Deployment awareness
  - Guardrails (safety limits & filters)
"""

import re
import hashlib
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# PII Masking / Data Anonymization
# ─────────────────────────────────────────────

class PIIMasker:
    """
    Data Anonymization / PII Masking:
    Strips out Personally Identifiable Information before sending data
    to third-party LLM APIs.
    """
    
    DEFAULT_PATTERNS = {
        "SSN": r'\b\d{3}-\d{2}-\d{4}\b',
        "CREDIT_CARD": r'\b(?:\d{4}[-\s]?){3}\d{4}\b',
        "EMAIL": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        "PHONE_US": r'\b(?:\+1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b',
        "IP_ADDRESS": r'\b(?:\d{1,3}\.){3}\d{1,3}\b',
        "DATE_OF_BIRTH": r'\b(?:0[1-9]|1[0-2])[/\-](?:0[1-9]|[12]\d|3[01])[/\-](?:19|20)\d{2}\b',
    }
    
    def __init__(self, extra_patterns: Optional[Dict[str, str]] = None):
        self.patterns = dict(self.DEFAULT_PATTERNS)
        if extra_patterns:
            self.patterns.update(extra_patterns)
        self._mask_map: Dict[str, str] = {}  # Store mappings for potential unmask
    
    def mask(self, text: str) -> str:
        """Replace all PII matches with masked tokens."""
        masked_text = text
        for pii_type, pattern in self.patterns.items():
            matches = re.findall(pattern, masked_text, re.IGNORECASE)
            for match in matches:
                token = f"[{pii_type}_MASKED_{hashlib.md5(match.encode()).hexdigest()[:6]}]"
                self._mask_map[token] = match
                masked_text = masked_text.replace(match, token)
                logger.info(f"PII Masking: Masked {pii_type} occurrence")
        return masked_text
    
    def unmask(self, text: str) -> str:
        """Restore masked tokens to original values (for internal use only)."""
        unmasked = text
        for token, original in self._mask_map.items():
            unmasked = unmasked.replace(token, original)
        return unmasked
    
    def get_mask_report(self) -> Dict[str, str]:
        """Return a report of all masked items."""
        return dict(self._mask_map)


# ─────────────────────────────────────────────
# Prompt Injection Defense
# ─────────────────────────────────────────────

class PromptInjectionDefense:
    """
    Prompt Injection Defense:
    Implements firewalls to catch malicious user prompts designed to
    hijack the agent's internal system instructions.
    """
    
    INJECTION_PATTERNS = [
        r'ignore\s+(?:previous\s+|above\s+|all\s+|prior\s+)*instructions?',
        r'disregard\s+(?:previous\s+|above\s+|all\s+|prior\s+)*instructions?',
        r'forget\s+(everything|all|previous|prior)',
        r'you\s+are\s+now\s+',
        r'new\s+instructions?\s*:',
        r'system\s*prompt\s*:',
        r'reveal\s+(your|the)\s+(system|internal|hidden)',
        r'override\s+(previous|system|all)',
        r'act\s+as\s+(if\s+)?(you\s+are\s+)?a\s+different',
        r'pretend\s+(you\s+are|to\s+be)',
        r'(print|show|display|output)\s+(your|the)\s+(system|initial)\s+(prompt|instructions?)',
        r'what\s+(is|are)\s+your\s+(system|initial)\s+(prompt|instructions?)',
    ]
    
    def __init__(self, extra_patterns: Optional[List[str]] = None):
        self.patterns = list(self.INJECTION_PATTERNS)
        if extra_patterns:
            self.patterns.extend(extra_patterns)
        self._compiled = [re.compile(p, re.IGNORECASE) for p in self.patterns]
    
    def check(self, prompt: str) -> Tuple[bool, Optional[str]]:
        """
        Check if a prompt contains injection attempts.
        Returns (is_safe, matched_pattern_description)
        """
        for i, pattern in enumerate(self._compiled):
            if pattern.search(prompt):
                description = self.patterns[i]
                logger.warning(f"Prompt Injection detected: pattern '{description}' matched")
                return False, f"Blocked by injection pattern: {description}"
        return True, None
    
    def sanitize(self, prompt: str) -> str:
        """Remove or neutralize injection attempts from the prompt."""
        sanitized = prompt
        for pattern in self._compiled:
            sanitized = pattern.sub("[BLOCKED_CONTENT]", sanitized)
        return sanitized


# ─────────────────────────────────────────────
# Role-Based Access Control (RBAC)
# ─────────────────────────────────────────────

@dataclass
class UserProfile:
    """Represents a user with RBAC attributes."""
    user_id: str
    username: str
    clearance_level: int = 0          # 0=public, 1=internal, 2=confidential, 3=restricted, 4=top_secret
    department: str = "general"
    roles: List[str] = field(default_factory=lambda: ["viewer"])
    allowed_collections: List[str] = field(default_factory=list)


class RBACManager:
    """
    Role-Based Access Control (RBAC):
    Restricts data retrieval based on user clearance.
    Ensures regular employees cannot retrieve restricted data.
    """
    
    CLEARANCE_LABELS = {
        0: "public",
        1: "internal",
        2: "confidential",
        3: "restricted",
        4: "top_secret",
    }
    
    def __init__(self):
        self.users: Dict[str, UserProfile] = {}
        self._register_default_user()
    
    def _register_default_user(self):
        """Register a default public user."""
        default = UserProfile(
            user_id="default",
            username="default_user",
            clearance_level=0,
            department="general",
            roles=["viewer"],
        )
        self.users["default"] = default
    
    def register_user(self, profile: UserProfile):
        """Register or update a user profile."""
        self.users[profile.user_id] = profile
        logger.info(f"RBAC: Registered user '{profile.username}' with clearance {profile.clearance_level}")
    
    def get_user(self, user_id: str) -> UserProfile:
        """Retrieve user profile."""
        return self.users.get(user_id, self.users["default"])
    
    def check_access(self, user_id: str, required_clearance: int) -> bool:
        """Check if user has sufficient clearance."""
        user = self.get_user(user_id)
        has_access = user.clearance_level >= required_clearance
        if not has_access:
            logger.warning(
                f"RBAC: Access denied for user '{user.username}' "
                f"(clearance {user.clearance_level}) to resource "
                f"requiring clearance {required_clearance}"
            )
        return has_access
    
    def get_metadata_filter(self, user_id: str) -> Dict:
        """
        Metadata Filtering:
        Generate a filter dict for vector DB queries based on user RBAC.
        """
        user = self.get_user(user_id)
        filter_dict = {
            "security_clearance": {"$lte": user.clearance_level},
        }
        if user.department != "general":
            filter_dict["department"] = {"$in": [user.department, "general"]}
        if user.allowed_collections:
            filter_dict["collection"] = {"$in": user.allowed_collections}
        return filter_dict


# ─────────────────────────────────────────────
# Guardrails (Safety Limits & Filters)
# ─────────────────────────────────────────────

class Guardrails:
    """
    Guardrails:
    Safety limits and filters to prevent the agent from performing
    unauthorized actions or generating harmful content.
    """
    
    # Actions that require HITL approval
    CRITICAL_ACTIONS = [
        "send_email",
        "execute_sql",
        "delete_data",
        "modify_permissions",
        "access_external_api",
        "write_file",
        "execute_code",
    ]
    
    # Content filters for output
    BLOCKED_CONTENT_PATTERNS = [
        r'(?i)(password|secret|token)\s*[:=]\s*\S+',
        r'(?i)api[_\s]?key\s*[:=]\s*\S+',
    ]
    
    def __init__(self, pii_masker: PIIMasker, injection_defense: PromptInjectionDefense,
                 rbac_manager: RBACManager):
        self.pii_masker = pii_masker
        self.injection_defense = injection_defense
        self.rbac_manager = rbac_manager
        self._blocked_patterns = [re.compile(p) for p in self.BLOCKED_CONTENT_PATTERNS]
    
    def process_input(self, text: str, user_id: str = "default") -> Tuple[str, Dict]:
        """
        Full input guardrail pipeline:
        1. Check prompt injection
        2. Mask PII
        3. Validate RBAC
        Returns (processed_text, report)
        """
        report = {
            "injection_safe": True,
            "pii_masked": False,
            "rbac_valid": True,
            "original_length": len(text),
        }
        
        # 1. Prompt injection defense
        is_safe, injection_msg = self.injection_defense.check(text)
        if not is_safe:
            report["injection_safe"] = False
            report["injection_message"] = injection_msg
            text = self.injection_defense.sanitize(text)
            logger.warning(f"Guardrails: Prompt injection detected and sanitized")
        
        # 2. PII masking
        masked_text = self.pii_masker.mask(text)
        if masked_text != text:
            report["pii_masked"] = True
            report["pii_items"] = len(self.pii_masker.get_mask_report())
            text = masked_text
        
        report["processed_length"] = len(text)
        return text, report
    
    def process_output(self, text: str) -> str:
        """
        Output guardrail: filter sensitive content from LLM responses.
        """
        filtered = text
        for pattern in self._blocked_patterns:
            filtered = pattern.sub("[REDACTED]", filtered)
        return filtered
    
    def requires_hitl(self, action: str) -> bool:
        """Check if an action requires Human-in-the-Loop approval."""
        return action in self.CRITICAL_ACTIONS
    
    def validate_action(self, action: str, user_id: str, required_clearance: int = 0) -> Tuple[bool, str]:
        """
        Full action validation:
        1. Check RBAC
        2. Check if HITL is needed
        Returns (allowed, reason)
        """
        # RBAC check
        if not self.rbac_manager.check_access(user_id, required_clearance):
            return False, f"Insufficient clearance for action '{action}'"
        
        # HITL check (returns True meaning 'needs approval', not 'blocked')
        if self.requires_hitl(action):
            return True, f"Action '{action}' requires Human-in-the-Loop approval"
        
        return True, "Action allowed"


# ─────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────

def create_security_stack(config=None) -> Guardrails:
    """Create the full security stack with default or custom config."""
    pii_masker = PIIMasker()
    injection_defense = PromptInjectionDefense()
    rbac_manager = RBACManager()
    return Guardrails(pii_masker, injection_defense, rbac_manager)
