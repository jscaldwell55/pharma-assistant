# backend/core_logic/knowledge_bridge.py
"""
Enables LLM to use limited, vetted pharmaceutical knowledge for query understanding
while maintaining strict safety boundaries.
"""

import re
from typing import Dict, List, Optional, Set, Tuple
import logging

logger = logging.getLogger("knowledge_bridge")

class PharmaceuticalKnowledgeBridge:
    """
    Provides safe, limited knowledge augmentation for pharmaceutical queries.
    This allows the system to understand synonyms and relationships without
    opening the door to hallucinations.
    """
    
    def __init__(self):
        # CRITICAL: Only add VERIFIED, SAFE mappings here
        # These should come from your approved documentation
        
        # Drug name mappings (brand <-> generic)
        self.drug_synonyms = {
            "zepbound": {"tirzepatide", "zepbound"},
            "tirzepatide": {"zepbound", "tirzepatide"},
            # Add common misspellings/variations
            "zep bound": {"tirzepatide", "zepbound"},
            "tizepatide": {"tirzepatide", "zepbound"},  # common typo
        }
        
        # Concept mappings for query understanding
        self.concept_mappings = {
            # Approval/indication synonyms
            "approved for": {"indicated for", "indication", "approved", "used for", "treats"},
            "indicated for": {"approved for", "indication", "approved", "used for", "treats"},
            "indication": {"approved for", "indicated for", "used for", "treats"},
            
            # Side effects synonyms
            "side effects": {"adverse reactions", "adverse events", "side effect", "reactions"},
            "adverse reactions": {"side effects", "adverse events", "side effect", "reactions"},
            "adverse events": {"side effects", "adverse reactions", "side effect", "reactions"},
            
            # Dosing synonyms
            "dosage": {"dose", "dosing", "how much", "amount", "strength"},
            "dose": {"dosage", "dosing", "how much", "amount", "strength"},
            "dosing": {"dosage", "dose", "how much", "amount", "strength"},
            
            # Administration synonyms
            "how to take": {"administration", "how to use", "inject", "taking"},
            "administration": {"how to take", "how to use", "inject", "taking"},
            
            # Storage synonyms
            "storage": {"store", "storing", "keep", "refrigerate"},
            "store": {"storage", "storing", "keep", "refrigerate"},
        }
        
        # Pharmaceutical context terms that should trigger expanded search
        self.context_triggers = {
            "weight loss", "weight management", "obesity", "diabetes", 
            "type 2 diabetes", "t2d", "glycemic control", "a1c"
        }
    
    def expand_query_safe(self, query: str) -> Tuple[str, List[str], Dict[str, any]]:
        """
        Safely expand query with verified synonyms and concepts.
        Returns: (enhanced_query, search_variants, metadata)
        """
        query_lower = query.lower()
        expansions = set()
        metadata = {
            "drug_detected": None,
            "concepts_detected": [],
            "expansions_applied": []
        }
        
        # 1. Detect and expand drug names
        for drug, synonyms in self.drug_synonyms.items():
            if drug in query_lower:
                expansions.update(synonyms)
                metadata["drug_detected"] = drug
                metadata["expansions_applied"].append(f"drug:{drug}")
                logger.info(f"Detected drug '{drug}', adding synonyms: {synonyms}")
        
        # 2. Detect and expand concepts
        for concept, synonyms in self.concept_mappings.items():
            if concept in query_lower:
                expansions.update(synonyms)
                metadata["concepts_detected"].append(concept)
                metadata["expansions_applied"].append(f"concept:{concept}")
        
        # 3. Build search variants
        search_variants = [query]  # Original query first
        
        # Add targeted variants if we found expansions
        if expansions:
            # Create a variant with key expansions
            expansion_terms = list(expansions)[:3]  # Limit to top 3 to avoid query explosion
            for term in expansion_terms:
                if term not in query_lower:
                    search_variants.append(f"{query} {term}")
        
        # 4. Create enhanced query for context building
        enhanced_query = query
        if metadata["drug_detected"] and "tirzepatide" not in query_lower and "zepbound" not in query_lower:
            # Help the LLM understand the drug context
            drug_context = f"(regarding {'/'.join(self.drug_synonyms[metadata['drug_detected']])})"
            enhanced_query = f"{query} {drug_context}"
        
        return enhanced_query, search_variants[:5], metadata  # Limit variants to prevent over-retrieval
    
    def create_context_preface(self, query: str, metadata: Dict[str, any]) -> Optional[str]:
        """
        Create a safe context preface that helps the LLM understand query intent
        without introducing external knowledge.
        """
        if not metadata.get("expansions_applied"):
            return None
        
        preface_parts = []
        
        if metadata.get("drug_detected"):
            drug = metadata["drug_detected"]
            synonyms = self.drug_synonyms.get(drug, set())
            if len(synonyms) > 1:
                preface_parts.append(
                    f"Note: The terms {' and '.join(sorted(synonyms))} refer to the same medication in this context."
                )
        
        if metadata.get("concepts_detected"):
            # Don't overwhelm with concept mappings, just acknowledge understanding
            preface_parts.append(
                "Query understood to be asking about: " + ", ".join(metadata["concepts_detected"])
            )
        
        return "\n".join(preface_parts) if preface_parts else None


class EnhancedRAGRetriever:
    """
    Enhanced retriever that uses the knowledge bridge for better recall
    while maintaining safety.
    """
    
    def __init__(self, base_retriever, knowledge_bridge: Optional[PharmaceuticalKnowledgeBridge] = None):
        self.base_retriever = base_retriever
        self.knowledge_bridge = knowledge_bridge or PharmaceuticalKnowledgeBridge()
    
    def retrieve(self, user_query: str) -> List[Dict[str, any]]:
        """
        Enhanced retrieval with safe query expansion.
        """
        # Get expanded query and variants
        enhanced_query, search_variants, metadata = self.knowledge_bridge.expand_query_safe(user_query)
        
        logger.info(f"Enhanced retrieval - Original: '{user_query}'")
        logger.info(f"Search variants: {search_variants}")
        logger.info(f"Metadata: {metadata}")
        
        # Use the base retriever's multi-variant search capability
        # by temporarily injecting our variants
        original_method = self.base_retriever._build_variants
        
        def enhanced_variants(q: str, intent: Optional[str]) -> Tuple[List[str], List[str]]:
            base_variants, key_terms = original_method(q, intent)
            # Merge our knowledge-based variants with RAG's intent-based ones
            combined_variants = list(dict.fromkeys(search_variants + base_variants))  # Remove duplicates
            
            # Add expansion terms to key terms for term-rescue
            if metadata.get("drug_detected"):
                key_terms.extend(list(self.knowledge_bridge.drug_synonyms[metadata["drug_detected"]]))
            
            return combined_variants[:8], list(set(key_terms))  # Limit total variants
        
        # Temporarily replace the variant builder
        self.base_retriever._build_variants = enhanced_variants
        
        try:
            # Run retrieval with enhanced variants
            results = self.base_retriever.retrieve(user_query)
            
            # Attach metadata for downstream use
            for r in results:
                r["query_metadata"] = metadata
            
            return results
        finally:
            # Restore original method
            self.base_retriever._build_variants = original_method
    
    # Delegate all other methods to base retriever
    def __getattr__(self, name):
        """Pass through any other method calls to the base retriever."""
        return getattr(self.base_retriever, name)