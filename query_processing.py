"""
Query processing module: normalization, expansion, and rewriting.
Handles query preprocessing to improve retrieval quality.
"""

import re
import unicodedata
from typing import Dict, List, Optional

from config import (
    ABBREVIATION_MAP,
    ENABLE_QUERY_NORMALIZATION,
    ENABLE_QUERY_EXPANSION,
    MAX_QUERY_EXPANSIONS,
)


class QueryNormalizer:
    """Production-grade query normalization pipeline."""
    
    def __init__(self, abbreviations: Dict[str, str] = None):
        self.abbreviations = abbreviations or ABBREVIATION_MAP
        self._whitespace_re = re.compile(r'\s+')
        self._special_chars_re = re.compile(r'[^\w\s\-\'\"?.,!]')
        self._repeated_punct_re = re.compile(r'([?!.,])\1+')
        
    def normalize(self, query: str) -> Dict[str, any]:
        """Full normalization: unicode → whitespace → punctuation → abbreviations."""
        if not query or not query.strip():
            return {
                "original": query,
                "normalized": "",
                "display": "",
                "expansions": []
            }
        
        original = query
        expansions = []
        
        # Unicode normalize + clean
        text = unicodedata.normalize('NFKC', query).strip()
        text = self._whitespace_re.sub(' ', text)
        text = self._special_chars_re.sub('', text)
        text = self._repeated_punct_re.sub(r'\1', text)
        
        display = text
        text_lower = text.lower()
        
        # Expand abbreviations
        words = text_lower.split()
        expanded = []
        for word in words:
            word_clean = word.rstrip('?.,!"\'-')
            trailing = word[len(word_clean):] if len(word_clean) < len(word) else ''
            if word_clean in self.abbreviations:
                exp = self.abbreviations[word_clean]
                expansions.append(f"{word_clean} → {exp}")
                expanded.append(exp + trailing)
            else:
                expanded.append(word)
        
        return {
            "original": original,
            "normalized": ' '.join(expanded),
            "display": display,
            "expansions": expansions,
        }
    
    def quick_normalize(self, query: str) -> str:
        """Quick normalization returning only the normalized string."""
        return self.normalize(query)["normalized"]


class QueryExpander:
    """Generates semantic query variants for improved recall."""
    
    PATTERNS = [
        (r'^what is (the )?(.*?)\??$', ['define {0}', 'explain {0}', '{0} definition']),
        (r'^how (do|does|can|to) (.*?)\??$', ['method for {0}', 'process of {0}']),
        (r'^why (is|are|do|does) (.*?)\??$', ['reason for {0}', 'cause of {0}']),
    ]
    
    STOP_WORDS = {
        'what', 'is', 'are', 'the', 'a', 'an', 'how', 'do', 'does', 'can',
        'why', 'when', 'where', 'which', 'who', 'to', 'for', 'in', 'on',
        'at', 'by', 'with', 'about', 'of', 'from', 'this', 'that', 'it',
        'be', 'been', 'being', 'have', 'has', 'had', 'and', 'or', 'but',
    }
    
    def __init__(self, max_expansions: int = MAX_QUERY_EXPANSIONS):
        self.max_expansions = max_expansions
        
    def expand(self, query: str) -> List[str]:
        """Generate query variants for improved recall."""
        variants = [query]
        query_lower = query.lower().strip()
        
        # Pattern-based expansion
        for pattern, templates in self.PATTERNS:
            match = re.match(pattern, query_lower, re.IGNORECASE)
            if match:
                content = match.groups()[-1]
                for tmpl in templates[:self.max_expansions]:
                    variant = tmpl.format(content)
                    if variant not in variants:
                        variants.append(variant)
                break
        
        # Keyword extraction variant
        keywords = self._extract_keywords(query)
        if keywords and keywords not in variants:
            variants.append(keywords)
        
        return variants[:self.max_expansions + 1]
    
    def _extract_keywords(self, query: str) -> str:
        """Extract meaningful keywords from query."""
        words = re.findall(r'\b\w+\b', query.lower())
        keywords = [w for w in words if w not in self.STOP_WORDS and len(w) > 2]
        return ' '.join(keywords) if keywords else ''


class QueryProcessor:
    """Main query processing orchestrator."""
    
    def __init__(
        self,
        enable_normalization: bool = ENABLE_QUERY_NORMALIZATION,
        enable_expansion: bool = ENABLE_QUERY_EXPANSION,
    ):
        self.normalizer = QueryNormalizer()
        self.expander = QueryExpander()
        self.enable_normalization = enable_normalization
        self.enable_expansion = enable_expansion
    
    def process(self, query: str) -> Dict:
        """Process query: normalize → expand."""
        result = {
            "original": query,
            "normalized": query,
            "variants": [query],
            "expansions": []
        }
        
        if self.enable_normalization:
            norm = self.normalizer.normalize(query)
            result["normalized"] = norm["normalized"]
            result["expansions"] = norm["expansions"]
        
        if self.enable_expansion:
            result["variants"] = self.expander.expand(result["normalized"])
        
        return result


# Global instance
query_processor = QueryProcessor()

