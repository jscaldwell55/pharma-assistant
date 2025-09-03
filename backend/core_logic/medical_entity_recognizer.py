# medical_entity_recognizer.py - Medical Entity Recognition System

import logging
import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

# Since scispacy requires large models, we'll create a lightweight alternative
# that can be upgraded to scispacy later

class EntityType(Enum):
    DRUG = "drug"
    DOSAGE = "dosage"
    CONDITION = "condition"
    SYMPTOM = "symptom"
    ANATOMY = "anatomy"
    PROCEDURE = "procedure"
    FREQUENCY = "frequency"
    ROUTE = "route"  # administration route

@dataclass
class MedicalEntity:
    text: str
    entity_type: EntityType
    start_pos: int
    end_pos: int
    confidence: float = 1.0
    normalized_form: Optional[str] = None

class MedicalEntityRecognizer:
    """
    Lightweight medical entity recognition system
    Can be upgraded to use scispacy: pip install scispacy
    """
    
    def __init__(self, use_scispacy: bool = False):
        self.use_scispacy = use_scispacy
        self.nlp = None
        
        if use_scispacy:
            try:
                import scispacy
                import spacy
                # Download with: pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.1/en_core_sci_md-0.5.1.tar.gz
                self.nlp = spacy.load("en_core_sci_md")
                logger.info("Loaded scispacy model for medical NER")
            except Exception as e:
                logger.warning(f"Could not load scispacy, using pattern matching: {e}")
                self.use_scispacy = False
        
        # Fallback pattern-based recognition
        self._init_medical_patterns()
    
    def _init_medical_patterns(self):
        """Initialize medical terminology patterns"""
        
        # Common drugs (extend this list based on your PDFs)
        self.drug_patterns = [
            r"\bJournvax\b",
            r"\b(?:aspirin|ibuprofen|acetaminophen|metformin|lisinopril)\b",
            r"\b(?:atorvastatin|omeprazole|amoxicillin|prednisone)\b",
            # Add specific drugs from your documentation
        ]
        
        # Dosage patterns
        self.dosage_patterns = [
            r"\b\d+\s*(?:mg|mcg|g|ml|mL|IU|units?)\b",
            r"\b\d+\.\d+\s*(?:mg|mcg|g|ml|mL|IU|units?)\b",
            r"\b(?:once|twice|three times|four times)\s+(?:a day|daily|per day)\b",
            r"\b(?:every|q)\s*\d+\s*(?:hours?|hrs?|days?|weeks?)\b",
        ]
        
        # Medical conditions
        self.condition_patterns = [
            r"\b(?:diabetes|hypertension|depression|anxiety|asthma)\b",
            r"\b(?:COPD|CHF|MI|CVA|TIA|DVT|PE)\b",
            r"\b(?:cancer|carcinoma|lymphoma|leukemia)\b",
            r"\b(?:infection|inflammation|disease|disorder|syndrome)\b",
        ]
        
        # Symptoms
        self.symptom_patterns = [
            r"\b(?:pain|headache|nausea|vomiting|dizziness)\b",
            r"\b(?:fever|chills|fatigue|weakness|malaise)\b",
            r"\b(?:rash|itching|swelling|redness|bruising)\b",
            r"\b(?:shortness of breath|dyspnea|cough|wheeze)\b",
        ]
        
        # Administration routes
        self.route_patterns = [
            r"\b(?:oral|PO|by mouth|sublingual)\b",
            r"\b(?:intravenous|IV|injection|subcutaneous|SC|IM)\b",
            r"\b(?:topical|transdermal|inhaled|nasal)\b",
        ]
        
        # Anatomy
        self.anatomy_patterns = [
            r"\b(?:heart|liver|kidney|lung|brain|stomach)\b",
            r"\b(?:blood|bone|muscle|skin|nerve)\b",
            r"\b(?:cardiovascular|respiratory|digestive|nervous)\s+system\b",
        ]
    
    def extract_entities(self, text: str) -> List[MedicalEntity]:
        """Extract medical entities from text"""
        
        if self.use_scispacy and self.nlp:
            return self._extract_with_scispacy(text)
        else:
            return self._extract_with_patterns(text)
    
    def _extract_with_scispacy(self, text: str) -> List[MedicalEntity]:
        """Extract entities using scispacy"""
        entities = []
        
        doc = self.nlp(text)
        
        for ent in doc.ents:
            entity_type = self._map_scispacy_label(ent.label_)
            if entity_type:
                entities.append(MedicalEntity(
                    text=ent.text,
                    entity_type=entity_type,
                    start_pos=ent.start_char,
                    end_pos=ent.end_char,
                    confidence=0.9  # Scispacy is pretty good
                ))
        
        return entities
    
    def _map_scispacy_label(self, label: str) -> Optional[EntityType]:
        """Map scispacy labels to our entity types"""
        mapping = {
            "DRUG": EntityType.DRUG,
            "CHEMICAL": EntityType.DRUG,
            "DISEASE": EntityType.CONDITION,
            "SYMPTOM": EntityType.SYMPTOM,
            "ANATOMY": EntityType.ANATOMY,
            "PROCEDURE": EntityType.PROCEDURE,
        }
        return mapping.get(label)
    
    def _extract_with_patterns(self, text: str) -> List[MedicalEntity]:
        """Extract entities using regex patterns"""
        entities = []
        
        # Extract drugs
        for pattern in self.drug_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                entities.append(MedicalEntity(
                    text=match.group(),
                    entity_type=EntityType.DRUG,
                    start_pos=match.start(),
                    end_pos=match.end(),
                    confidence=0.8
                ))
        
        # Extract dosages
        for pattern in self.dosage_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                entities.append(MedicalEntity(
                    text=match.group(),
                    entity_type=EntityType.DOSAGE,
                    start_pos=match.start(),
                    end_pos=match.end(),
                    confidence=0.9
                ))
        
        # Extract conditions
        for pattern in self.condition_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                entities.append(MedicalEntity(
                    text=match.group(),
                    entity_type=EntityType.CONDITION,
                    start_pos=match.start(),
                    end_pos=match.end(),
                    confidence=0.7
                ))
        
        # Extract symptoms
        for pattern in self.symptom_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                entities.append(MedicalEntity(
                    text=match.group(),
                    entity_type=EntityType.SYMPTOM,
                    start_pos=match.start(),
                    end_pos=match.end(),
                    confidence=0.7
                ))
        
        # Extract routes
        for pattern in self.route_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                entities.append(MedicalEntity(
                    text=match.group(),
                    entity_type=EntityType.ROUTE,
                    start_pos=match.start(),
                    end_pos=match.end(),
                    confidence=0.8
                ))
        
        # Remove duplicates
        entities = self._deduplicate_entities(entities)
        
        return entities
    
    def _deduplicate_entities(self, entities: List[MedicalEntity]) -> List[MedicalEntity]:
        """Remove overlapping entities, keeping highest confidence"""
        if not entities:
            return entities
        
        # Sort by position and confidence
        entities.sort(key=lambda e: (e.start_pos, -e.confidence))
        
        deduplicated = []
        last_end = -1
        
        for entity in entities:
            if entity.start_pos >= last_end:
                deduplicated.append(entity)
                last_end = entity.end_pos
        
        return deduplicated
    
    def analyze_query_intent(self, text: str, entities: List[MedicalEntity]) -> Dict[str, any]:
        """Analyze query intent based on entities"""
        
        intent = {
            "has_drug_mention": any(e.entity_type == EntityType.DRUG for e in entities),
            "has_dosage_query": any(e.entity_type == EntityType.DOSAGE for e in entities),
            "has_condition": any(e.entity_type == EntityType.CONDITION for e in entities),
            "has_symptom": any(e.entity_type == EntityType.SYMPTOM for e in entities),
            "primary_drug": None,
            "query_type": None
        }
        
        # Find primary drug if mentioned
        drug_entities = [e for e in entities if e.entity_type == EntityType.DRUG]
        if drug_entities:
            intent["primary_drug"] = drug_entities[0].text
        
        # Determine query type
        text_lower = text.lower()
        if "side effect" in text_lower or "adverse" in text_lower:
            intent["query_type"] = "side_effects"
        elif "dose" in text_lower or "dosage" in text_lower or "how much" in text_lower:
            intent["query_type"] = "dosage"
        elif "interact" in text_lower or "with other" in text_lower:
            intent["query_type"] = "interactions"
        elif "storage" in text_lower or "store" in text_lower:
            intent["query_type"] = "storage"
        elif "contraindication" in text_lower or "should not" in text_lower:
            intent["query_type"] = "contraindications"
        else:
            intent["query_type"] = "general"
        
        return intent

# Singleton instance
_recognizer_instance: Optional[MedicalEntityRecognizer] = None

def get_medical_recognizer(use_scispacy: bool = False) -> MedicalEntityRecognizer:
    """Get singleton medical entity recognizer"""
    global _recognizer_instance
    if _recognizer_instance is None:
        _recognizer_instance = MedicalEntityRecognizer(use_scispacy=use_scispacy)
    return _recognizer_instance