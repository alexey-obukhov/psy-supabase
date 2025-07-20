"""
This module provides centralized mappings for therapeutic concepts,
including themes, keywords, emotions, and their interrelationships.
It is designed to ensure consistency in how various psychological
constructs are identified, categorized, and utilized across the application,
particularly in topic detection, emotion analysis, and response generation.

The `TherapeuticMappings` class serves as a static container for these
mappings, offering methods to retrieve specific data points like keywords
for a theme, a therapeutic approach for a given theme, or a response
template for a particular approach. This centralization helps in maintaining
and updating the knowledge base of the application in a structured manner.
"""

import re
from typing import Any, Dict, Final, List, Optional, Tuple, Union

from psy_supabase import get_package_logger

logger = get_package_logger(__name__)


class TherapeuticMappings:
    """
    Centralized repository for therapeutic mappings.

    This class holds various dictionaries and provides class methods
    to access and interpret them. These mappings are crucial for:
    - Identifying psychological themes from user input.
    - Detecting emotions and their intensity.
    - Mapping themes to appropriate therapeutic approaches.
    - Mapping approaches to specific response templates.

    Attributes:
        ENHANCED_TAXONOMY (Dict[str, List[str]]):
            A detailed taxonomy mapping broad psychological categories (e.g., "depression",
            "anxiety", "workplace_trauma") to extensive lists of related keywords and phrases.
            This is primarily used for keyword-based topic spotting and understanding
            the nuances within a category.

        THERAPEUTIC_THEMES (Dict[str, Dict[str, Any]]):
            Maps core therapeutic themes (e.g., "trauma", "anxiety", "grief_loss",
            "workplace_anxiety") to their associated keywords, common emotions,
            and suggested therapeutic approaches. This helps in a more holistic
            understanding of a user's expressed issue.

        APPROACH_TO_TEMPLATE (Dict[str, str]):
            Maps specific therapeutic approaches (e.g., "cbt", "trauma_informed",
            "supportive_listening") to corresponding response template names. This
            enables the system to select an appropriate conversational flow or
            response structure based on the identified therapeutic strategy.

        EMOTION_PATTERNS (Dict[str, List[Tuple[str, float]]]):
            A comprehensive collection of regular expression patterns for detecting
            various emotions and their intensity (weights). This was the original
            combined list. (Legacy, see CORE_EMOTIONS, etc.)

        CORE_EMOTIONS (Dict[str, List[Tuple[str, float]]]):
            Defines regex patterns and weights specifically for detecting fundamental
            affect states or "pure" emotions like anger, fear, sadness, joy.

        CLINICAL_PATTERNS (Dict[str, List[Tuple[str, float]]]):
            Defines regex patterns and weights for detecting terms and phrases
            associated with clinical or diagnostic categories like anxiety disorders,
            depression, trauma, and stress.

        INTERPERSONAL_PATTERNS (Dict[str, List[Tuple[str, float]]]):
            Defines regex patterns and weights for detecting themes related to
            interpersonal relationships (e.g., "relationships", "loneliness") and
            self-concept (e.g., "self-esteem", "identity").
    """

    ENHANCED_TAXONOMY: Dict[str, List[str]] = {
        "depression": [
            "melancholy",
            "fatigue",
            "tired",
            "unmotivated",
            "don't enjoy",
            "insomnia",
            "exhausted",
            "weight",
            "can't sleep",
            "appetite",
            "guilt",
            "concentration",
            "depressed",
            "no motivation",
            "unhappy",
            "empty",
            "suicidal",
            "no energy",
            "no interest",
            "lost interest",
            "sad",
            "hypersomnia",
            "indecisive",
            "numb",
            "anhedonia",
            "can't eat",
            "psychomotor",
            "can't enjoy",
            "worthless",
            "gloomy",
            "despair",
            "miserable",
            "emptiness",
            "meaningless",
            "depression",
            "hopeless",
        ],
        "anxiety": [
            "nervous",
            "on edge",
            "frightened",
            "catastrophizing",
            "dread",
            "arousal",
            "apprehensive",
            "overwhelmed",
            "anxious",
            "racing thoughts",
            "uneasy",
            "hypervigilant",
            "stress",
            "panic",
            "worry",
            "social anxiety",
            "scared",
            "obsessive",
            "avoidance",
            "tense",
            "irritable",
            "fear",
            "restless",
            "compulsive",
            "phobia",
            "worried",
            "anxiety",
            "overthinking",
            "performance anxiety",
        ],
        "trauma": [
            "neglect",
            "avoidance",
            "nightmare",
            "trauma",
            "violence",
            "hyperarousal",
            "harass",
            "ptsd",
            "assaulted",
            "startle",
            "horror",
            "childhood trauma",
            "threat",
            "helpless",
            "assault",
            "intrusion",
            "flashback",
            "emotional dysregulation",
            "survivor",
            "accident",
            "danger",
            "numb",
            "triggered",
            "traumatized",
            "dissociate",
            "disaster",
            "victim",
            "abuse",
            "hypervigilant",
            "abused",
            "victimized",
            "detached",
        ],
        "control": [
            "control",
            "controlling",
            "loss of control",
            "need control",
            "powerless",
            "micromanage",
            "manipulate",
            "dominate",
            "overpower",
            "helpless to stop",
            "can't manage",
            "grip",
            "handle",
            "let go",  # Also related to control (or lack thereof)
            "surrender",  # Related to letting go of control
            "accept",  # Related to accepting lack of control
            "influence",
            "authority",
            "discipline",
            "order",
            "structure",
            "boundaries",  # Can be related to control
            "freedom",  # Lack of control or freedom from control
            "autonomy",
            "independence",
            "dependence",  # Can be related to control dynamics
            "compulsion",  # Feeling out of control
            "obsession",  # Can be about controlling thoughts/actions
            "perfectionism",
        ],
        # workplace trauma and abuse
        "workplace_trauma": [
            # Primary workplace abuse terms (stronger matches)
            "workplace abuse",
            "work abuse",
            "boss abuse",
            "manager abuse",
            "toxic workplace",
            "hostile work",
            "bullied at work",
            "harassed at work",
            "workplace harassment",
            "workplace bullying",
            "abused at work",
            "work trauma",
            "workplace trauma",
            "toxic boss",
            "toxic manager",
            "abusive supervisor",
            "boss bully",
            "manager bully",
            "workplace bully",
            # Secondary workplace terms
            "mobbing",
            "work stress",
            "threatened at work",
            "intimidated at work",
            "humiliated at work",
            "workplace retaliation",
            "work mistreatment",
            "gaslighting at work",
            "workplace injustice",
            "unfair treatment at work",
            # Additional workplace problem indicators
            "discriminated at work",
            "work discrimination",
            "hostile environment",
            "career sabotage",
            "workplace violence",
            "demotion",
            "unfair review",
            "fired unfairly",
            "targeted at work",
            "work anxiety",
            "job trauma",
            "toxic team",
            "toxic coworker",
            "work harassment",
            "work bullying",
            # Common phrases
            "hate my job",
            "hate my boss",
            "terrible workplace",
            "awful job",
            "hostile boss",
            "mean coworker",
            "being bullied",
            "being harassed",
            "work is hell",
            "office politics",
            "power abuse",
            "authority abuse",
            "work ptsd",
            "verbally abused",
            "yelled at",
            "screamed at",
        ],
        # Relationship-related topics
        "relationship": [
            "relationship",
            "marriage",
            "partner",
            "boyfriend",
            "girlfriend",
            "husband",
            "wife",
            "spouse",
            "couple",
            "dating",
            "significant other",
            "ex",
            "breakup",
            "divorce",
            "separated",
            "together",
            "commitment",
            "trust",
            "betrayal",
            "cheating",
            "infidelity",
            "jealousy",
            "communication",
            "argument",
            "fight",
            "romantic",
            "love",
            "loved",
            "loving",
            "connection",
            "attachment",
        ],
        # heartbreak and healing
        "heartbreak": [
            "heartbreak",
            "heartbroken",
            "broken heart",
            "broken up",
            "dumped",
            "rejected",
            "betrayed",
            "abandoned",
            "alone",
            "lonely",
            "miss them",
            "missing them",
            "moving on",
            "get over",
            "heal",
            "healing",
            "closure",
            "broken heart",
            "love pain",
            "hurt by love",
            "hurt by them",
            "never again",
            "brake my heart",
            "break my heart",
            "no more love",
            "trust again",
            "never trust",
            "fall in love",
            "falling for someone",
            "vulnerable",
        ],
        "interpersonal": [
            "relationship",
            "marriage",
            "partner",
            "spouse",
            "family",
            "friend",
            "colleague",
            "conflict",
            "intimacy",
            "attachment",
            "boundary",
            "communication",
            "trust",
            "abandonment",
            "rejection",
            "loneliness",
            "isolation",
            "connection",
            "breakup",
            "divorce",
            "separation",
            "betrayal",
            "argument",
            "misunderstanding",
        ],
        "identity": [
            "self-esteem",
            "identity",
            "self-worth",
            "confidence",
            "imposter",
            "shame",
            "perfectionism",
            "failure",
            "inadequacy",
            "self-doubt",
            "body image",
            "self-criticism",
            "self-compassion",
            "validation",
            "purpose",
            "meaning",
            "values",
            "authentic",
            "true self",
            "gender",
            "sexuality",
            "culture",
        ],
        "adjustment": [
            "grief",
            "loss",
            "bereavement",
            "change",
            "transition",
            "adaptation",
            "adjustment",
            "stress",
            "coping",
            "resilience",
            "life stage",
            "retirement",
            "career",
            "moving",
            "relocation",
            "major life event",
            "crisis",
            "upheaval",
            "uncertainty",
            "decision-making",
            "crossroads",
            "opportunity",
            "challenge",
        ],
        "behavior": [
            "addiction",
            "substance",
            "alcohol",
            "drug",
            "gambling",
            "compulsive",
            "habit",
            "dependence",
            "withdrawal",
            "craving",
            "relapse",
            "recovery",
            "abstinence",
            "moderation",
            "harm-reduction",
            "impulse control",
            "self-regulation",
            "behavioral therapy",
            "reinforcement",
            "trigger",
        ],
        "wellness": [
            "mindfulness",
            "meditation",
            "relaxation",
            "self-care",
            "resilience",
            "growth",
            "strength",
            "resource",
            "wellness",
            "prevention",
            "maintenance",
            "balance",
            "harmony",
            "fulfillment",
            "joy",
            "satisfaction",
            "gratitude",
            "meaning",
            "purpose",
            "flourishing",
            "thriving",
            "vitality",
        ],
        "cognition": [
            "thought",
            "belief",
            "cognition",
            "distortion",
            "schema",
            "assumption",
            "automatic thought",
            "rumination",
            "worry",
            "attention",
            "memory",
            "concentration",
            "problem-solving",
            "decision-making",
            "perception",
            "interpretation",
            "reframe",
            "perspective",
            "mindset",
            "attribution",
        ],
        "grief_loss": [
            "bereavement",
            "loss",
            "mourning",
            "acceptance",
            "denial",
            "anger",
            "bargaining",
            "depression",
            "adaptation",
            "adjustment",
            "memorialization",
            "letting go",
            "moving on",
            "honoring",
            "memory",
        ],
        "self-compassion": [
            "self-kindness",
            "common humanity",
            "mindfulness",
            "self-criticism",
            "self-care",
            "forgiveness",
            "acceptance",
            "compassionate voice",
            "inner peace",
            "empathy",
        ],
        "guilt": [
            "guilt",
            "guilty",
            "regret",
            "wrongdoing",
            "redemption",
            "forgiveness",
            "moral distress",
            "self-forgiveness",
            "conscience",
            "remorse",
            "apologize",
            "sorry",
        ],
        "shame": [
            "shame",
            "ashamed",
            "humiliation",
            "embarrassment",
            "mortified",
            "self-conscious",
            "inadequate",
            "unworthy",
            "defective",
            "flawed",
            "exposed",
            "inferior",
            "self-judgment",
            "self-blame",
        ],
        "obsessive_compulsive_disorder": [
            "obsession",
            "compulsion",
            "ritual",
            "perfectionism",
            "control",
            "anxiety",
            "reassurance-seeking",
            "intrusive thought",
            "cleaning",
            "checking",
            "counting",
            "hoarding",
        ],
        "suicidality_self_harm": [
            "suicidal",
            "self-harm",
            "cutting",
            "despair",
            "hopelessness",
            "crisis",
            "emotional pain",
            "coping",
            "prevention",
            "life-threatening",
            "overwhelming",
        ],
        "emotional_support": ["help", "support", "understand", "listen", "care", "concern"],
        "family_dynamics": [
            # Core family terms
            "family dynamics",
            "family issues",
            "family problems",
            "family relationship",
            "family tension",
            "family conflict",
            "family drama",
            "family situation",
            "family stress",
            # Parent-child relationships
            "parent child",
            "parental",
            "parents",
            "mother",
            "father",
            "mom",
            "dad",
            "parent relationship",
            "parenting style",
            "strict parents",
            "controlling parents",
            "distant parents",
            # Sibling relationships
            "sibling rivalry",
            "brother",
            "sister",
            "siblings",
            "favorite child",
            "golden child",
            "scapegoat",
            # Extended family
            "grandparent",
            "aunt",
            "uncle",
            "cousin",
            "in-laws",
            "extended family",
            "family gathering",
            # Family roles and patterns
            "family role",
            "black sheep",
            "peacemaker",
            "mediator",
            "caretaker role",
            "parentification",
            "family rules",
            "family expectations",
            "family pressure",
            # Childhood experiences
            "childhood",
            "growing up",
            "upbringing",
            "raised",
            "family history",
            "family background",
            "childhood memories",
            "childhood trauma",
            "childhood experiences",
            # Family behaviors
            "favoritism",
            "rejection",
            "comparison",
            "criticism",
            "approval seeking",
            "validation seeking",
            "emotional neglect",
            "conditional love",
            "family boundaries",
            "family communication",
            # Family emotions
            "family disappointment",
            "family shame",
            "family guilt",
            "family pride",
            "family loyalty",
            "family obligation",
            "family resentment",
            "family jealousy",
            "family anger",
        ],
        "childhood_issues": [
            # Core childhood terms
            "childhood issues",
            "childhood problems",
            "childhood trauma",
            "early experiences",
            "growing up",
            "young age",
            # Development periods
            "early childhood",
            "middle childhood",
            "adolescence",
            "teenage years",
            "youth",
            "developmental",
            # Childhood environments
            "home environment",
            "school experiences",
            "neighborhood",
            "community",
            "cultural background",
            "religious upbringing",
            # Childhood relationships
            "childhood friends",
            "peer relationships",
            "bullying",
            "social inclusion",
            "social exclusion",
            "friendship issues",
            # Educational experiences
            "school problems",
            "academic pressure",
            "learning difficulties",
            "school anxiety",
            "teacher relationships",
            "education stress",
        ],
        "approval_seeking": [
            # Core approval terms
            "need approval",
            "seeking approval",
            "validation seeking",
            "people pleasing",
            "perfectionism",
            "fear of rejection",
            # Behavioral patterns
            "trying to please",
            "cant say no",
            "overachiever",
            "perfectionist",
            "overcompensating",
            "prove myself",
            # Related emotions
            "fear of disappointment",
            "fear of criticism",
            "fear of judgment",
            "need to be perfect",
            "fear of failure",
            "fear of abandonment",
        ],
        "favoritism": [
            # Core favoritism terms - word stems that catch variations
            "favor",
            "favors",
            "favorite",
            "favorites",
            "favoritism",
            "favored",
            "favoring",
            "prefer",
            "prefers",
            "preferred",
            "preference",
            "preferential",
            "preferring",
            # Family favoritism phrases
            "golden child",
            "scapegoat",
            "black sheep",
            "favorite child",
            "preferred child",
            "special treatment",
            "treated differently",
            "different treatment",
            "unfair treatment",
            "unequal treatment",
            "double standards",
            # Parental favoritism
            "mom's favorite",
            "dad's favorite",
            "mother's favorite",
            "father's favorite",
            "parents prefer",
            "clearly prefers",
            "obviously favors",
            "always chooses",
            "mommy's boy",
            "daddy's girl",
            "mama's boy",
            "papa's girl",
            # Comparative language
            "loves more",
            "cares more",
            "pays more attention",
            "spends more time",
            "gives more to",
            "shows more love",
            "more affectionate with",
            "always takes their side",
            "takes his side",
            "takes her side",
            # Sibling competition
            "sibling rivalry",
            "sibling comparison",
            "compared to sibling",
            "brother gets",
            "sister gets",
            "sibling gets everything",
            "they get everything",
            "gets away with everything",
            "can do no wrong",
            "never gets in trouble",
            "always gets what they want",
            # Emotional impact phrases
            "never measure up",
            "always second best",
            "not as good as",
            "not the favorite",
            "less important",
            "valued less",
            "feeling left out",
            "feeling inferior",
            "feeling less than",
            "jealous of sibling",
            "disappointed parents",
            "not good enough for them",
            # Behavioral indicators
            "special privileges",
            "gets special attention",
            "gets better treatment",
            "why can't you be like",
            "your sibling would",
            "they never",
            "living in shadow",
            "compete with",
            "always chosen",
            "never picked",
            # Family dynamics
            "parental preference",
            "family competition",
            "plays favorites",
            "pick favorites",
            "choosing favorites",
            "clear favorite",
            "obvious favorite",
            "always liked better",
            "treat better",
            "loved more than",
        ],
        "family": [
            # Core family terms
            "family",
            "families",
            "familial",
            "parent",
            "parents",
            "parental",
            "mother",
            "father",
            "mom",
            "dad",
            "sibling",
            "siblings",
            "brother",
            "sister",
            # Extended family
            "grandparent",
            "aunt",
            "uncle",
            "cousin",
            "in-law",
            "in-laws",
            "extended family",
            # Family relationships
            "family relationship",
            "family dynamic",
            "family issue",
            "family problem",
            "family conflict",
            "family tension",
            # Common phrases
            "my family",
            "our family",
            "the family",
            "family member",
            "family situation",
            "within the family",
            "in my family",
        ],
        "approval": [
            # Direct terms
            "approval",
            "approve",
            "approved",
            "validation",
            "validate",
            "validated",
            "acceptance",
            "accept",
            "accepted",
            # Seeking patterns
            "need approval",
            "seeking approval",
            "want approval",
            "looking for approval",
            "need validation",
            "seeking validation",
            "want validation",
            "looking for validation",
            # Common phrases
            "want them to approve",
            "makes them happy",
            "please others",
            "make proud",
            "live up to",
            "meet expectations",
            "prove myself",
            "gain acceptance",
            "earn love",
        ],
        "rejection": [
            # Direct terms
            "reject",
            "rejected",
            "rejection",
            "exclude",
            "excluded",
            "exclusion",
            "abandon",
            "abandoned",
            "abandonment",
            # Feelings and experiences
            "left out",
            "pushed away",
            "not wanted",
            "unwanted",
            "cast aside",
            "pushed aside",
            "ignored",
            "overlooked",
            "dismissed",
            "shut out",
            "not included",
            "don't belong",
            "doesn't want me",
            "don't want me",
            # Family-specific
            "family rejection",
            "parental rejection",
            "sibling rejection",
            "rejected by family",
            "family abandonment",
        ],
        "childhood": [
            # Direct terms
            "child",
            "childhood",
            "children",
            "kid",
            "kids",
            "young",
            # Time periods
            "growing up",
            "grew up",
            "when I was young",
            "as a child",
            "as kids",
            "early years",
            "younger years",
            # Experiences
            "raised",
            "upbringing",
            "childhood experience",
            "childhood memory",
            "childhood trauma",
            # Family context
            "family history",
            "family background",
            "family upbringing",
            "childhood home",
            "childhood family",
        ],
        "insecurity": [
            # Core insecurity terms
            "insecure",
            "insecurity",
            "insecurities",
            "uncertain",
            "uncertainty",
            "doubt",
            "doubts",
            "doubting",
            "self-doubt",
            "self-conscious",
            "inadequate",
            "inadequacy",
            "not enough",
            "not good enough",
            "vulnerable",
            "vulnerability",
            "exposed",
            "fragile",
            # Relationship insecurity
            "relationship insecurity",
            "insecure in relationship",
            "feel insecure",
            "afraid of losing",
            "fear of abandonment",
            "fear losing them",
            "not secure in relationship",
            "worried about relationship",
            "relationship anxiety",
            "attachment anxiety",
            "clingy",
            # Self-worth related
            "low self-esteem",
            "poor self-image",
            "lack confidence",
            "don't feel worthy",
            "unworthy",
            "not deserving",
            "feel small",
            "feel insignificant",
            "feel lesser",
            "compare myself",
            "comparing myself",
            "not measuring up",
            # Physical/appearance insecurity
            "body insecurity",
            "appearance anxiety",
            "how I look",
            "ugly",
            "unattractive",
            "fat",
            "skinny",
            "too short",
            "too tall",
            "hate my body",
            "don't like how I look",
            "self-image issues",
            # Performance insecurity
            "imposter syndrome",
            "feel like fraud",
            "don't belong",
            "afraid of failure",
            "fear of judgment",
            "what others think",
            "not smart enough",
            "not talented enough",
            "out of my league",
        ],
        "jealousy": [
            # Core jealousy terms
            "jealous",
            "jealousy",
            "envious",
            "envy",
            "resentful",
            "resentment",
            "green with envy",
            "bitter",
            "possessive",
            "territorial",
            # Relationship jealousy
            "jealous of partner",
            "partner talking to",
            "worried about cheating",
            "suspicious",
            "don't trust",
            "checking phone",
            "following",
            "stalking",
            "monitoring",
            "watching",
            "spying",
            # Social jealousy
            "jealous of friends",
            "jealous of siblings",
            "jealous of coworkers",
            "they have everything",
            "why can't I have",
            "wish I had",
            "not fair they get",
            "they don't deserve",
            "I deserve more",
            # Success/achievement jealousy
            "jealous of success",
            "envious of achievements",
            "why them not me",
            "they got promoted",
            "they have better",
            "more successful than me",
            "everyone else has",
            "left behind",
            "missing out",
            # Emotional expressions
            "makes me sick",
            "burns me up",
            "can't stand seeing",
            "hate when they",
            "bothers me when",
            "upsets me that",
        ],
        "trust": [
            # Core trust terms
            "trust",
            "trusted",
            "trusting",
            "trustworthy",
            "distrust",
            "mistrust",
            "faith",
            "confidence",
            "belief",
            "rely",
            "relying",
            "dependable",
            # Trust issues
            "trust issues",
            "hard to trust",
            "don't trust",
            "can't trust",
            "lost trust",
            "broken trust",
            "betrayed",
            "betrayal",
            "lied to",
            "cheated on",
            "let down",
            "disappointed",
            # Relationship trust
            "trust in relationship",
            "trust my partner",
            "faithful",
            "loyalty",
            "being honest",
            "telling truth",
            "keeping secrets",
            "hiding things",
            "suspicious behavior",
            "acting strange",
            "something's wrong",
            # Building/rebuilding trust
            "learning to trust",
            "want to trust",
            "trying to trust",
            "rebuild trust",
            "regain trust",
            "earn trust back",
            "prove trustworthy",
            "show I can trust",
            "give another chance",
            # Past trauma affecting trust
            "burned before",
            "hurt in past",
            "past relationships",
            "previous betrayal",
            "once bitten twice shy",
            "walls up",
            "guard up",
            "protective",
        ],
        # Also enhance existing relationship theme
        "relationship": [
            # Existing relationship terms plus additions for insecurity patterns
            "relationship",
            "relationships",
            "partner",
            "boyfriend",
            "girlfriend",
            "spouse",
            "husband",
            "wife",
            "dating",
            "couple",
            "romantic",
            # Relationship problems
            "relationship problems",
            "relationship issues",
            "relationship trouble",
            "fighting",
            "arguing",
            "conflict",
            "tension",
            "distance",
            "growing apart",
            "drifting apart",
            "disconnected",
            # Relationship insecurity patterns
            "needy",
            "clingy",
            "possessive",
            "controlling",
            "demanding",
            "checking up on",
            "need constant reassurance",
            "seeking validation",
            "afraid they'll leave",
            "fear of being alone",
            "abandonment",
            # Communication issues
            "don't communicate",
            "can't talk",
            "won't listen",
            "misunderstand",
            "not hearing me",
            "don't feel heard",
            "ignored",
            "dismissed",
            # Commitment issues
            "commitment",
            "committed",
            "exclusive",
            "serious",
            "casual",
            "moving too fast",
            "moving too slow",
            "ready for next step",
            "marriage",
            "engagement",
            "living together",
            "future together",
        ],
        "humiliation": [
            # Core humiliation terms
            "humiliate",
            "humiliated",
            "humiliating",
            "humiliation",
            "embarrass",
            "embarrassed",
            "embarrassing",
            "embarrassment",
            "mortify",
            "mortified",
            "mortifying",
            "mortification",
            "shame",
            "shamed",
            "shaming",
            "ashamed",
            # Workplace humiliation
            "publicly humiliated",
            "humiliated at work",
            "embarrassed at work",
            "made to look stupid",
            "made fun of",
            "laughed at",
            "mocked",
            "ridiculed",
            "belittled",
            "put down",
            "degraded",
            "demeaned",
            # Social humiliation
            "humiliated in front of",
            "embarrassed in public",
            "made a fool of",
            "looked like an idiot",
            "felt stupid",
            "felt foolish",
            "lost face",
            "dignity stripped",
            "pride wounded",
            # Emotional impact
            "want to disappear",
            "crawl under a rock",
            "hide my face",
            "never show my face",
            "die of embarrassment",
            "mortified beyond belief",
            "crushed my spirit",
            "destroyed my confidence",
            "shattered my ego",
            # Professional humiliation
            "dress down",
            "dressed down",
            "chewed out",
            "torn apart",
            "ripped to shreds",
            "destroyed in meeting",
            "called out publicly",
            "made example of",
            "singled out",
            "targeted for criticism",
        ],
        "inadequacy": [
            # Core inadequacy terms
            "inadequate",
            "inadequacy",
            "not enough",
            "not good enough",
            "insufficient",
            "deficient",
            "lacking",
            "falling short",
            "subpar",
            "below standard",
            "not up to par",
            "not measuring up",
            # Self-perception of inadequacy
            "feel inadequate",
            "feeling inadequate",
            "sense of inadequacy",
            "not capable",
            "incapable",
            "incompetent",
            "not qualified",
            "out of my depth",
            "in over my head",
            "can't handle it",
            "not cut out for",
            "don't have what it takes",
            "not skilled enough",
            # Comparison-based inadequacy
            "everyone else is better",
            "others are more capable",
            "can't compete",
            "behind everyone else",
            "not as good as others",
            "lagging behind",
            "can't keep up",
            "struggling to keep up",
            "outclassed",
            # Professional inadequacy
            "not qualified for job",
            "imposter syndrome",
            "fake it till you make it",
            "don't belong here",
            "hired by mistake",
            "over my head at work",
            "can't do the job",
            "failing at work",
            "not meeting expectations",
            # Emotional expressions
            "feel like a failure",
            "total failure",
            "complete failure",
            "disappointment",
            "let everyone down",
            "not worthy",
            "don't deserve",
            "undeserving",
            "not earned my place",
        ],
        "workplace": [
            # Core workplace terms
            "work",
            "workplace",
            "job",
            "office",
            "career",
            "professional",
            "employment",
            "employer",
            "employee",
            "staff",
            "team",
            # Workplace roles
            "boss",
            "manager",
            "supervisor",
            "coworker",
            "colleague",
            "subordinate",
            "executive",
            "leadership",
            "management",
            "hr",
            "human resources",
            "department",
            "company",
            "organization",
            # Work activities
            "meeting",
            "presentation",
            "project",
            "deadline",
            "task",
            "assignment",
            "responsibility",
            "performance",
            "evaluation",
            "review",
            "feedback",
            "promotion",
            "raise",
            "bonus",
            # Work environment
            "office culture",
            "work environment",
            "corporate",
            "professional setting",
            "work atmosphere",
            "team dynamics",
            "office politics",
            "work relationships",
            "professional relationships",
            # Work-related stress
            "work stress",
            "job stress",
            "workplace pressure",
            "work pressure",
            "work anxiety",
            "job anxiety",
            "career stress",
            "professional stress",
            "work-life balance",
            "overwork",
            "overtime",
            "workload",
            # Work problems
            "work issues",
            "job problems",
            "workplace problems",
            "work conflict",
            "workplace conflict",
            "work drama",
            "office drama",
            "work troubles",
        ],
        "criticism": [
            # Core criticism terms
            "criticize",
            "criticized",
            "criticizing",
            "criticism",
            "critique",
            "judge",
            "judged",
            "judging",
            "judgment",
            "judgmental",
            "blame",
            "blamed",
            "blaming",
            "fault",
            "faulted",
            "faulting",
            # Types of criticism
            "harsh criticism",
            "constant criticism",
            "unfair criticism",
            "constructive criticism",
            "destructive criticism",
            "brutal criticism",
            "nitpicking",
            "fault-finding",
            "picking apart",
            "tearing down",
            # Receiving criticism
            "being criticized",
            "under criticism",
            "criticized for",
            "picked on",
            "singled out",
            "targeted",
            "attacked",
            "condemned",
            "denounced",
            "censured",
            "reprimanded",
            # Self-criticism
            "self-criticism",
            "self-critical",
            "critical of myself",
            "hard on myself",
            "my own worst critic",
            "beat myself up",
            "self-blame",
            "blame myself",
            "fault myself",
            # Workplace criticism
            "criticized at work",
            "boss criticizes",
            "manager criticizes",
            "performance criticism",
            "work criticism",
            "professional criticism",
            "negative feedback",
            "poor evaluation",
            "bad review",
            # Emotional impact of criticism
            "can't take criticism",
            "sensitive to criticism",
            "hurt by criticism",
            "crushed by criticism",
            "destroyed by criticism",
            "devastated by feedback",
            "feel attacked",
            "feel judged",
            "feel condemned",
            # Family/relationship criticism
            "criticized by family",
            "parents criticize",
            "criticized by partner",
            "constant judgment",
            "never good enough",
            "always finding fault",
        ],
        # Also enhance the existing workplace_trauma section with more specific terms
        "workplace_trauma": [
            # Existing terms plus new ones
            "workplace abuse",
            "work abuse",
            "boss abuse",
            "manager abuse",
            "toxic workplace",
            "hostile work",
            "bullied at work",
            "harassed at work",
            "workplace harassment",
            "workplace bullying",
            "abused at work",
            "work trauma",
            "workplace trauma",
            "toxic boss",
            "toxic manager",
            # Humiliation-specific workplace trauma
            "humiliated at work",
            "embarrassed at work",
            "publicly shamed",
            "made example of",
            "singled out",
            "called out publicly",
            "criticized in front of others",
            "torn apart in meeting",
            # Inadequacy-inducing workplace trauma
            "made to feel stupid",
            "told I'm incompetent",
            "questioned my abilities",
            "undermined my confidence",
            "made to feel small",
            "belittled my work",
            "dismissed my ideas",
            "ignored my contributions",
            "overlooked for promotion",
            # Professional sabotage
            "sabotaged my work",
            "set me up to fail",
            "impossible deadlines",
            "unrealistic expectations",
            "moving goalposts",
            "changing requirements",
            "no support",
            "thrown under the bus",
            "scapegoated",
            # Power abuse
            "abuse of power",
            "authority abuse",
            "position abuse",
            "rank abuse",
            "threatened my job",
            "intimidation tactics",
            "retaliation",
            "punitive measures",
            "disciplinary action",
            "write-ups",
            # Emotional workplace abuse
            "gaslighting at work",
            "mind games",
            "psychological manipulation",
            "emotional abuse",
            "verbal abuse",
            "yelling",
            "screaming",
            "condescending",
            "patronizing",
            "talking down to",
        ],
    }

    THERAPEUTIC_THEMES: Dict[str, Dict[str, Any]] = {
        # Trauma & PTSD
        "trauma": {
            "keywords": ["trauma", "flashback", "ptsd", "abuse", "childhood_issues"],
            "emotions": ["fear", "anxiety", "helplessness"],
            "approaches": ["trauma", "cognitive_behavioral"],
        },
        # Anxiety
        "anxiety": {
            "keywords": ["anxiety", "worry", "stress", "panic", "nervous", "overthinking", "health_anxiety"],
            "emotions": ["anxiety", "fear", "tension"],
            "approaches": ["cognitive_behavioral", "mindfulness"],
        },
        "control": {
            "keywords": [
                "control",
                "controlling",
                "loss of control",
                "powerless",
                "manage",
                "grip",
                "handle",
                "micromanage",
                "let go",
                "can't stop",
                "need to control",
                "out of control",
            ],
            "emotions": ["anxiety", "frustration", "helplessness", "anger", "fear"],  # Example emotions
            "approaches": [
                "cognitive_behavioral",
                "acceptance_commitment",
                "mindfulness_relaxation",
                "dialectical_behavior",
            ],  # Example approaches
            "description": "Issues related to the need to control, feeling out of control, or being controlled.",
            "human_readable_name": "Control and Empowerment",
        },
        # Depression
        "depression": {
            "keywords": [
                "depression",
                "depressed",
                "sad",
                "hopeless",
                "unmotivated",
                "empty",
                "loss of interest",
                "exhausted",
                "no point",
                "anhedonia",
            ],
            "emotions": ["sadness", "hopelessness", "fatigue", "emptiness", "guilt"],
            "approaches": ["behavioral_activation", "cognitive_behavioral", "self_compassion", "motivation_support"],
            "description": "Persistent feelings of sadness, loss of interest, and other mood-related symptoms.",
            "human_readable_name": "Support for Depression",
        },
        # Grief & Loss
        "grief_loss": {
            "keywords": [
                "grief",
                "loss",
                "grief_loss",
                "death",
                "bereavement",
                "mourning",
                "passed away",
                "deceased",
                "gone",
                "passing",
                "died",
                "lost someone",
                "missing someone",
                "funeral",
                "memorial",
            ],
            "emotions": ["sadness", "longing", "emptiness"],
            "approaches": ["grief_processing", "supportive_listening"],
        },
        # Workplace Anxiety
        "workplace_anxiety": {
            "keywords": [
                "work",
                "job",
                "career",
                "boss",
                "workplace",
                "office",
                "work_stress",
                "workplace_stress",
                "workplace_trauma",
                "panic at work",
                "anxious about work",
            ],
            "emotions": ["anxiety", "stress", "frustration", "fear"],
            "approaches": ["cognitive_behavioral", "stress_management"],
            "description": "Anxiety and stress related to the workplace, job, or career.",
            "human_readable_name": "Workplace Anxiety Support",
        },
        # General Work Support
        "work": {
            "keywords": ["work", "job", "career", "boss", "office", "employment", "colleague"],
            "emotions": ["stress", "anxiety", "frustration", "pressure"],
            "approaches": ["cognitive_behavioral", "stress_management", "solution_focused"],
            "description": "General issues related to work, career, and the professional environment.",
            "human_readable_name": "Work-Related Support",
        },
        # Relationships
        "relationship_issues": {
            "keywords": [
                "relationship",
                "partner",
                "breakup",
                "marriage",
                "divorce",
                "dating",
                "couple",
                "romantic",
                "interpersonal",
                "family_conflict",
                "jealousy",
            ],
            "emotions": ["hurt", "confusion", "loneliness"],
            "approaches": ["interpersonal_therapy", "attachment_based"],
        },
        # General Emotional Support
        "general_support": {
            "keywords": ["support", "help", "listen", "understand", "talk", "struggling", "stuck"],
            "emotions": ["sadness", "distress", "neutral"],
            "approaches": ["self_compassion", "supportive_listening", "motivation_support"],
            "description": "General need for support, understanding, and a compassionate presence.",
            "human_readable_name": "General Emotional Support",
        },
        # Self-Compassion
        "self_compassion": {
            "keywords": [
                "self_doubt",
                "worthlessness",
                "insecurity",
                "impostor_syndrome",
                "not good enough",
                "approval_seeking",
                "self-criticism",
                "hard on myself",
            ],
            "emotions": ["shame", "inadequacy", "self-criticism"],
            "approaches": ["compassion_focused_therapy", "cognitive_behavioral"],
            "description": "Developing self-kindness and understanding towards oneself.",
            "human_readable_name": "Self-Compassion Building",
        },
    }

    APPROACH_TO_TEMPLATE: Dict[str, str] = {
        # Core emotions and conditions
        "anxiety": "anxiety",
        "depression": "depression",
        "loneliness": "loneliness",
        "trauma": "trauma",
        "grief": "grief_loss",
        "shame": "shame",
        "guilt": "guilt",
        "heartbreak": "heartbreak",
        "control": "control",
        # Control-related variations
        "controlling": "control",
        "need_control": "control",
        "loss_of_control": "control",
        "out_of_control": "control",
        # Heartbreak-related variations
        "broken_heart": "heartbreak",
        "breakup": "heartbreak",
        "relationship_loss": "heartbreak",
        "love_loss": "heartbreak",
        # Add connection-building to loneliness mapping
        "connection_building": "loneliness",
        "connection": "loneliness",
        "social_connection": "loneliness",
        "isolation": "loneliness",
        # Core therapeutic approaches
        "cbt": "cognitive_behavioral_therapy",
        "cognitive_behavioral": "cognitive_behavioral_therapy",
        "cognitive behavioral": "cognitive_behavioral_therapy",
        "cognitive-behavioral": "cognitive_behavioral_therapy",
        "behavioral_activation": "depression",  # Maps approach to a theme-named template
        "behavioral activation": "depression",
        "interpersonal_therapy": "relationship_issues",
        "interpersonal therapy": "relationship_issues",
        "ipt": "relationship_issues",
        # Acceptance and mindfulness approaches
        "act": "acceptance_commitment_therapy",
        "acceptance": "acceptance_commitment_therapy",
        "acceptance_commitment": "acceptance_commitment_therapy",
        "acceptance and commitment": "acceptance_commitment_therapy",
        "mindfulness": "mindfulness_relaxation",
        "mindfulness_relaxation": "mindfulness_relaxation",
        "meditation": "mindfulness_relaxation",
        # Grief and loss approaches
        "grief_loss": "grief_loss",
        "grief loss": "grief_loss",
        "grief-loss": "grief_loss",
        "grief_processing": "grief_loss",
        "grief_reflection": "grief_loss",
        "bereavement": "grief_loss",
        "loss": "grief_loss",
        "mourning": "grief_loss",
        # Empathy, validation and compassion (unified)
        "empathy": "empathy_validation",
        "validation": "empathy_validation",
        "supportive": "empathy_validation",
        "supportive_listening": "empathy_validation",
        "emotional_support": "empathy_validation",
        "general": "empathy_validation",
        "general_support": "empathy_validation",
        "listening": "empathy_validation",
        "understanding": "empathy_validation",
        "compassion": "empathy_validation",
        # Compassion-focused therapy (specific)
        "self-compassion": "self_compassion",
        "self_compassion": "self_compassion",
        "cft": "self_compassion",
        "compassion_focused": "self_compassion",
        "compassion-focused": "self_compassion",
        "compassion_focused_therapy": "self_compassion",
        "self-criticism": "self_compassion",
        "inner_kindness": "self_compassion",
        "inner_critic": "self_compassion",
        # Solution-focused approaches
        "solution": "solution_focused_brief_therapy",
        "solution_focused": "solution_focused_brief_therapy",
        "solution-focused": "solution_focused_brief_therapy",
        "sfbt": "solution_focused_brief_therapy",
        "brief_therapy": "solution_focused_brief_therapy",
        "goal_focused": "solution_focused_brief_therapy",
        "abandonment": "attachment_based_therapy",
        # Information and education
        "information": "information",
        "education": "information",
        "psychoeducation": "information",
        "explain": "information",
        "clarify": "information",
        "learn": "information",
        # DBT and variants
        "dbt": "dialectical_behavior_therapy",
        "dialectical": "dialectical_behavior_therapy",
        "dialectical_behavioral": "dialectical_behavior_therapy",
        "dialectical_behavior": "dialectical_behavior_therapy",
        # Workplace-related (maps to a specific template, not a general theme here)
        "workplace": "workplace_anxiety",  # Assuming workplace_anxiety.j2 template
        "work": "workplace_anxiety",
        "career": "workplace_anxiety",
        "job": "workplace_anxiety",
        "workplace_trauma": "workplace_anxiety",
        "workplace_stress": "workplace_anxiety",
        "work_stress": "workplace_anxiety",
        "workplace_anxiety": "workplace_anxiety",
        "work_anxiety": "workplace_anxiety",
        # Specialized conditions
        "ocd": "obsessive_compulsive_disorder",
        "obsessive": "obsessive_compulsive_disorder",
        "compulsive": "obsessive_compulsive_disorder",
        "obsessive_compulsive": "obsessive_compulsive_disorder",
        # Crisis and support
        "crisis": "crisis_support",
        "emergency": "crisis_support",
        "urgent": "crisis_support",
        "immediate": "crisis_support",
        "suicidal": "crisis_support",
        "crisis_intervention": "crisis_support",
        "suicidality": "suicidality_self_harm",
        "suicidality_self_harm": "suicidality_self_harm",
        # Motivational Interviewing approaches
        "motivational": "motivational_interviewing",
        "motivational_interviewing": "motivational_interviewing",
        "mi": "motivational_interviewing",
        "change_talk": "motivational_interviewing",
        "ambivalence": "motivational_interviewing",
        "readiness": "motivational_interviewing",
        # General Motivation Support
        "motivation_support": "motivation_support",
        "procrastination": "motivation_support",
        "feeling_stuck": "motivation_support",
        "unmotivated": "motivation_support",
        "lack_of_motivation": "motivation_support",
        "get_started": "motivation_support",
        "task_initiation": "motivation_support",
        # Mindfulness and relaxation
        "identity": "identity",
        "adjustment": "adjustment",
        "behavior": "behavior",
        "wellness": "wellness",
        "cognition": "cognition",
        "emotional_regulation": "emotional_regulation",
        "self-worth": "self-worth",
        "relationship": "relationship_issues",
        "interpersonal": "interpersonal_therapy",
        "self-esteem": "self_compassion",
        "trauma_informed": "trauma",
        "stress_management": "stress_management",
        "attachment_based": "attachment_based_therapy",
        # General fallback for "issues"
        "issues": "general_support",
    }

    EMOTION_PATTERNS: Final[Dict[str, List[Tuple[str, float]]]] = {
        "sadness": [
            (r"\bsad(ness)?\b", 2.0),
            (r"\bdown\b", 2.0),
            (r"\bunhappy\b", 1.5),
            (r"\bempty\b", 1.8),
            (r"\btear(s|ful)?\b", 1.5),
            (r"\bcry(ing)?\b", 1.5),
            (r"\bgrief\b", 2.0),
            (r"\bloss\b", 1.5),
            (r"\bheartbroken\b", 2.0),
            (r"\bmiserable\b", 1.5),
            (r"\bdepressed\b", 2.0),
            (r"\bhopeless\b", 1.5),
            (r"\bworthless\b", 1.5),
        ],
        "shame": [
            (r"\bashamed\b", 2.2),
            (r"\bshame\b", 2.0),
            (r"\bhumiliated\b", 1.8),
            (r"\bmortified\b", 1.8),
            (r"\bself-conscious\b", 1.5),
            (r"\binadequate\b", 1.5),
            (r"\bflawed\b", 1.5),
            (r"\bdefective\b", 1.5),
        ],
        "guilt": [
            (r"\bguilt(y)?\b", 2.0),
            (r"\bremorse\b", 1.8),
            (r"\bregret\b", 1.8),
            (r"\bsorry\b", 1.5),
            (r"\bapologetic\b", 1.5),
            (r"\bwrongdoing\b", 1.5),
            (r"\bself[- ]?blame\b", 1.5),
            (r"\bembarrass(ed|ment)?\b", 1.2),
        ],
        "anxiety": [
            # Standard patterns - with higher weight (2.0)
            (r"\banxiety\b", 2.0),
            (r"\banxious\b", 2.0),
            (r"\bpanic\b", 2.0),
            (r"\bworried\b", 1.8),
            (r"\bfear\b", 1.8),
            (r"\bstress(ed)?\b", 1.5),
            (r"\boverwhelm(ed|ing)\b", 1.5),
            # Subtle patterns
            (r"\bwhat (might|could) go wrong\b", 2.3),
            (r"\bconstantly worry\b", 1.5),
            (r"\bnervous\b", 1.5),
            (r"\bon edge\b", 1.5),
            (r"\bcan\'t relax\b", 1.5),
            (r"\brestless\b", 1.2),
            (r"\buneasy\b", 1.2),
            (r"\btense\b", 1.2),
            (r"\bworry about\b", 1.8),
            (r"\banxious about\b", 2.0),
        ],
        "depression": [
            # Standard patterns
            (r"\bdepress(ed|ion)\b", 2.0),
            (r"\bsad\b", 1.8),
            (r"\blow\b", 1.2),
            (r"\bmood\b", 1.0),
            (r"\bhopeless\b", 1.8),
            (r"\bunmotivated\b", 1.5),
            (r"\bexhausted\b", 1.2),
            # Subtle patterns
            (r"\bdon\'t enjoy\b", 1.5),
            (r"\bno pleasure\b", 1.5),
            (r"\blost interest\b", 1.5),
            (r"\bno energy\b", 1.2),
            (r"\bfeel empty\b", 1.5),
            (r"\bworthless\b", 1.8),
            (r"\btired all the time\b", 1.2),
            (r"\bfeeling down\b", 1.5),
        ],
        "trauma": [
            (r"\btrauma\b", 2.0),
            (r"\bptsd\b", 2.0),
            (r"\babuse\b", 1.8),
            (r"\bviolent\b", 1.5),
            (r"\bassault\b", 1.8),
            (r"\bincident\b", 1.0),
            (r"\bflashbacks\b", 1.8),
            (r"\bnightmares\b", 1.5),
            (r"\bhaunt\b", 1.2),
        ],
        "relationships": [
            (r"\bpartner\b", 1.5),
            (r"\bspouse\b", 1.5),
            (r"\bmarriage\b", 1.8),
            (r"\brelationship\b", 2.0),
            (r"\bdating\b", 1.5),
            (r"\bcouple\b", 1.2),
            (r"\bex\b", 1.3),
            (r"\bbreak[- ]?up\b", 1.8),
            (r"\bgirlfriend\b", 1.5),
            (r"\bboyfriend\b", 1.5),
            (r"\bhusband\b", 1.5),
            (r"\bwife\b", 1.5),
            (r"\bdivorce\b", 1.8),
            (r"\bseparation\b", 1.5),
        ],
        "self-esteem": [
            (r"\bself[- ]esteem\b", 2.0),
            (r"\bconfidence\b", 1.8),
            (r"\bworth\b", 1.5),
            (r"\bunlovable\b", 1.8),
            (r"\bunattractive\b", 1.5),
            (r"\binadequate\b", 1.8),
            (r"\bnot good enough\b", 2.0),
            (r"\bnever feel good enough\b", 2.0),
            (r"\bfailure\b", 1.8),
            (r"\bworthless\b", 1.8),
            (r"\bhate myself\b", 2.0),
            (r"\bugly\b", 1.5),
            (r"\bcompared to others\b", 1.8),
        ],
        "stress": [
            (r"\bstress(ed)?\b", 1.8),
            (r"\boverwhelm(ed|ing)\b", 1.8),
            (r"\bbusy\b", 1.0),
            (r"\bworkload\b", 1.5),
            (r"\bburn[- ]?out\b", 1.8),
            (r"\bcoping\b", 1.2),
            (r"\btoo much to do\b", 1.5),
            (r"\boverworked\b", 1.8),
        ],
        "identity": [
            (r"\bidentity\b", 1.8),
            (r"\bwho am I\b", 1.8),
            (r"\bmeaning\b", 1.5),
            (r"\bpurpose\b", 1.5),
            (r"\bdirection\b", 1.2),
            (r"\blife purpose\b", 1.8),
            (r"\bexistential\b", 1.8),
        ],
        "loneliness": [
            (r"\blonely\b", 2.0),
            (r"\balone\b", 1.8),
            (r"\bisolat(ed|ion)\b", 1.8),
            (r"\bno friends\b", 2.0),
            (r"\bsocially\b", 1.0),
            (r"\bconnection\b", 1.2),
            (r"\bno one\b", 1.5),
            (r"\bsolitude\b", 1.5),
        ],
        "fear": [
            (r"\bfear(ful)?\b", 2.0),
            (r"\bscared\b", 2.0),
            (r"\bafraid\b", 2.0),
            (r"\bterrified\b", 2.0),
            (r"\banxious\b", 2.0),
            (r"\banxiety\b", 2.0),
            (r"\bnervous\b", 1.5),
            (r"\bpanic\b", 2.0),
            (r"\bworry(ing)?\b", 1.5),
            (r"\boverwhelm(ed|ing)?\b", 1.5),
            (r"\buneasy\b", 1.2),
            (r"\btense\b", 1.2),
            (r"\bwon\'t work out\b", 1.8),
        ],
        "anger": [
            (r"\bangry\b", 2.0),
            (r"\banger\b", 2.0),
            (r"\bfrustrat(ed|ion)?\b", 1.8),
            (r"\bfurious\b", 2.5),
            (r"\birritat(ed|ion)?\b", 1.5),
            (r"\brage\b", 2.0),
            (r"\bmad\b", 1.5),
            (r"\bannoy(ed|ing)?\b", 1.2),
            (r"\bresent(ment)?\b", 1.5),
            (r"\bfed up\b", 2.3),
        ],
        "joy": [
            (r"\bhappy\b", 2.0),
            (r"\bjoy(ful)?\b", 2.0),
            (r"\bthrilled\b", 2.5),
            (r"\bgood news\b", 2.0),
            (r"\bexcited\b", 1.8),
            (r"\bcontent\b", 1.2),
            (r"\bgrateful\b", 1.2),
            (r"\bpleased\b", 1.2),
            (r"\boptimistic\b", 1.2),
            (r"\bhopeful\b", 1.2),
        ],
        "love": [
            (r"\blove\b", 2.0),
            (r"\bloved\b", 2.0),
            (r"\bcared\b", 1.5),
            (r"\bcaring\b", 1.2),
            (r"\baffection\b", 1.2),
            (r"\bconnected\b", 1.2),
            (r"\bclose\b", 1.2),
        ],
        "disgust": [
            (r"\bdisgust(ed|ing)?\b", 2.0),
            (r"\brepuls(ed|ion)?\b", 1.5),
            (r"\baversion\b", 1.2),
        ],
        "surprise": [
            (r"\bsurpris(ed|ing)?\b", 2.5),
            (r"\bhow things turned out\b", 2.0),
            (r"\bshocked\b", 1.5),
            (r"\bamazed\b", 1.2),
            (r"\bstunned\b", 1.2),
        ],
        "confusion": [
            (r"\bconfus(ed|ion)?\b", 2.0),
            (r"\buncertain\b", 1.5),
            (r"\bunsure\b", 1.5),
            (r"\bdoubt\b", 1.2),
        ],
        "relief": [
            (r"\brelief\b", 2.0),
            (r"\brelieved\b", 2.0),
            (r"\bcalm\b", 1.5),
            (r"\bsoothe(d|ing)?\b", 1.2),
            (r"\bpeaceful\b", 1.2),
        ],
    }

    # 1. PURE EMOTIONS - fundamental affect states
    CORE_EMOTIONS: Final[Dict[str, List[Tuple[str, float]]]] = {
        "anger": [
            (r"\bangry\b", 2.0),
            (r"\banger\b", 2.0),
            (r"\bfrustrat(ed|ion)?\b", 1.8),
            (r"\bfurious\b", 2.5),
            (r"\birritat(ed|ion)?\b", 1.5),
            (r"\brage\b", 2.0),
            (r"\bmad\b", 1.5),
            (r"\bannoy(ed|ing)?\b", 1.2),
            (r"\bresent(ment)?\b", 1.5),
            (r"\bfed up\b", 2.3),
        ],
        "fear": [
            (r"\bfear(ful)?\b", 2.0),
            (r"\bscared\b", 2.0),
            (r"\bafraid\b", 2.0),
            (r"\bterrified\b", 2.5),
            (r"\bworried\b", 2.2),
            (r"\bnervous\b", 1.5),
            (r"\buneasy\b", 1.2),
            (r"\btense\b", 1.2),
            (r"\bwon\'t work out\b", 1.8),
        ],
        "sadness": [
            (r"\bsad(ness)?\b", 2.0),
            (r"\bdown\b", 2.2),
            (r"\bunhappy\b", 1.5),
            (r"\btear(s|ful)?\b", 1.5),
            (r"\bcry(ing)?\b", 1.5),
            (r"\bmiserable\b", 1.5),
            (r"\bheartbroken\b", 2.0),
            (r"\bempty inside\b", 2.3),
            (r"\bfeel(ing)? really down\b", 2.4),
        ],
        "shame": [
            (r"\b(feel)?(a?)shame(d)?\b", 2.5),
            (r"\bhumiliated\b", 1.8),
            (r"\bmortified\b", 1.8),
            (r"\bself-conscious\b", 1.5),
            (r"\binadequate\b", 1.5),
            (r"\bflawed\b", 1.5),
            (r"\bdefective\b", 1.5),
            (r"\bforgive myself\b", 2.4),
        ],
        "guilt": [
            (r"\bguilt(y)?\b", 2.0),
            (r"\bremorse\b", 1.5),
            (r"\bregret\b", 1.5),
            (r"\bsorry\b", 1.5),
            (r"\bapologetic\b", 1.5),
            (r"\bwrongdoing\b", 1.5),
            (r"\bself[- ]?blame\b", 1.5),
            (r"\bembarrass(ed|ment)?\b", 1.2),
        ],
        "frustration": [
            (r"\bfrustrat(ed|ing|ion)\b", 1.8),
            (r"\bannoy(ed|ing)\b", 1.5),
            (r"\birritat(ed|ing)\b", 1.5),
        ],
        "joy": [
            (r"\bhappy\b", 2.0),
            (r"\bjoy(ful)?\b", 2.0),
            (r"\bthrilled\b", 2.5),
            (r"\bgood news\b", 2.0),
            (r"\bexcited\b", 1.8),
            (r"\bcontent\b", 1.2),
            (r"\bgrateful\b", 1.2),
            (r"\bpleased\b", 1.2),
            (r"\boptimistic\b", 1.2),
            (r"\bhopeful\b", 1.2),
        ],
        "love": [
            (r"\blove\b", 2.0),
            (r"\bloved\b", 2.0),
            (r"\bcared\b", 1.5),
            (r"\bcaring\b", 1.2),
            (r"\baffection\b", 1.2),
            (r"\bconnected\b", 1.2),
        ],
        "disgust": [
            (r"\bdisgust(ed|ing)?\b", 2.0),
            (r"\brepuls(ed|ion)?\b", 1.5),
            (r"\baversion\b", 1.2),
        ],
        "surprise": [
            (r"\bsurpris(ed|ing)?\b", 2.2),
            (r"\bshocked\b", 1.5),
            (r"\bamazed\b", 1.2),
            (r"\bstunned\b", 1.2),
        ],
        "confusion": [
            (r"\bconfus(ed|ion)?\b", 2.0),
            (r"\buncertain\b", 1.5),
            (r"\bunsure\b", 1.5),
            (r"\bdoubt\b", 1.2),
        ],
        "relief": [
            (r"\brelief\b", 2.0),
            (r"\brelieved\b", 2.0),
            (r"\bcalm\b", 1.5),
            (r"\bsoothe(d|ing)?\b", 1.2),
            (r"\bpeaceful\b", 1.2),
        ],
        "neutral": [
            (r"\bneutral\b", 1.0),
            (r"\bneither good nor bad\b", 1.0),
            (r"\b(don't|do not) feel much\b", 1.0),
        ],
        "hurt": [
            (r"\bhurt\b", 2.0),
            (r"\bpain(ed)?\b", 1.5),
            (r"\bwound(ed)?\b", 1.5),
            (r"\baching\b", 1.2),
        ],
        "longing": [
            (r"\blong(ing)? for\b", 2.0),
            (r"\byearn(ing)?\b", 1.8),
            (r"\bmiss(ing)? (him|her|them|it|someone)\b", 1.5),  # More specific
            (r"\bcrave\b", 1.2),
        ],
        "tension": [
            (r"\bten(se|sion)\b", 2.0),
            (r"\bstrain(ed)?\b", 1.5),
            (r"\bkeyed up\b", 1.2),
            (r"\bfeel tight\b", 1.2),
        ],
    }

    # 2. CLINICAL PATTERNS - clinical/diagnostic categories
    CLINICAL_PATTERNS: Final[Dict[str, List[Tuple[str, float]]]] = {
        "anxiety": [
            (r"\banxiety\b", 2.2),
            (r"\banxious\b", 2.2),
            (r"\bpanic\b", 2.0),
            (r"\bworried\b", 1.8),
            (r"\bstress(ed)?\b", 1.5),
            (r"\boverwhelm(ed|ing)\b", 1.5),
            (r"\bconstantly worry\b", 1.5),
            (r"\bon edge\b", 1.5),
            (r"\bcan\'t relax\b", 1.5),
            (r"\brestless\b", 1.2),
        ],
        "depression": [
            (r"\bdepress(ed|ion)\b", 2.0),
            (r"\blow\b", 1.2),
            # (r"\bhopeless\b", 1.8),
            (r"\bunmotivated\b", 1.5),
            (r"\bexhausted\b", 1.2),
            (r"\bdon\'t enjoy\b", 1.5),
            (r"\bno pleasure\b", 1.5),
            (r"\blost interest\b", 1.5),
            (r"\bno energy\b", 1.2),
            (r"\bfeel empty\b", 1.5),
            (r"\bworthless\b", 1.8),
        ],
        "pressure": [
            (r"\bpressure(d)?\b", 1.8),
            (r"\bunder pressure\b", 2.0),
            (r"\bfeel the heat\b", 1.5),
            (r"\bdeadline pressure\b", 1.5),
        ],
        "helplessness": [
            (r"\bhelpless(ness)?\b", 2.0),
            (r"\bpowerless\b", 1.8),
            (r"\bcan't do anything\b", 1.5),
            (r"\bno control over\b", 1.5),
            (r"\bfeel stuck\b", 1.2),  # Can also indicate helplessness
        ],
        "hopelessness": [
            (r"\bhopeless(ness)?\b", 2.0),
            (r"\bdespair(ing)?\b", 1.8),
            (r"\bno hope\b", 1.8),
            (r"\bsee no way out\b", 1.5),
            (r"\bgiven up\b", 1.5),
            (r"\bwhat's the point\b", 1.5),
        ],
        "fatigue": [
            (r"\bfatigue(d)?\b", 2.0),
            (r"\btired all the time\b", 1.8),
            (r"\bworn out\b", 1.5),
            (r"\bletharg(y|ic)\b", 1.5),
            (r"\bno physical energy\b", 1.2),
        ],
        "distress": [
            (r"\bdistress(ed)?\b", 2.0),
            (r"\bupset\b", 1.5),  # General term
            (r"\btroubled\b", 1.2),
            (r"\bagitated\b", 1.5),
        ],
        "trauma": [
            (r"\btrauma\b", 2.0),
            (r"\bptsd\b", 2.0),
            (r"\babuse\b", 1.8),
            (r"\bviolent\b", 1.5),
            (r"\bassault\b", 1.8),
            (r"\bflashbacks\b", 1.8),
            (r"\bnightmares\b", 1.5),
            (r"\bhaunt\b", 1.2),
        ],
        "stress": [
            (r"\bstress(ed)?\b", 1.8),
            (r"\boverwhelm(ed|ing)\b", 1.8),
            (r"\bworkload\b", 1.5),
            (r"\bburn[- ]?out\b", 1.8),
            (r"\bcoping\b", 1.2),
            (r"\btoo much to do\b", 1.5),
            (r"\boverworked\b", 1.8),
        ],
    }

    # 3. INTERPERSONAL/IDENTITY PATTERNS - relationship and self-concept
    INTERPERSONAL_PATTERNS: Final[Dict[str, List[Tuple[str, float]]]] = {
        "relationships": [
            (r"\brelationship\b", 2.0),
            (r"\bpartner\b", 1.5),
            (r"\bspouse\b", 1.5),
            (r"\bmarriage\b", 1.8),
            (r"\bdating\b", 1.5),
            (r"\bcouple\b", 1.2),
            (r"\bbreak[- ]?up\b", 1.8),
            (r"\bdivorce\b", 1.8),
        ],
        "loneliness": [
            (r"\blonely\b", 2.0),
            (r"\balone\b", 1.8),
            (r"\bisolat(ed|ion)\b", 1.8),
            (r"\bno friends\b", 2.0),
            (r"\bconnection\b", 1.2),
            (r"\bno one\b", 1.5),
            (r"\bsolitude\b", 1.5),
        ],
        "self-esteem": [
            (r"\bself[- ]esteem\b", 2.0),
            (r"\bconfidence\b", 1.8),
            (r"\bworth\b", 1.5),
            (r"\bunlovable\b", 1.8),
            (r"\bfailure\b", 2.2),
            (r"\bworthless\b", 2.2),
            (r"\bunattractive\b", 1.5),
            (r"\binadequate\b", 1.8),
            (r"\bnot good enough\b", 2.0),
            (r"\bnever feel good enough\b", 2.0),
            (r"\bhate myself\b", 2.0),
            (r"\bugly\b", 1.5),
            (r"\bcompared to others\b", 1.8),
        ],
        "identity": [
            (r"\bidentity\b", 1.8),
            (r"\bwho am I\b", 1.8),
            (r"\bmeaning\b", 1.5),
            (r"\bpurpose\b", 1.5),
            (r"\bdirection\b", 1.2),
            (r"\blife purpose\b", 1.8),
            (r"\bexistential\b", 1.8),
        ],
        "emptiness": [
            (r"\bempty(ness)?\b", 2.0),
            (r"\bvoid\b", 1.5),
            (r"\bhollow\b", 1.5),
            (r"\bfeel(ing)? nothing\b", 1.8),
        ],
        "inadequacy": [
            (r"\binadequate\b", 2.0),
            (r"\bnot good enough\b", 1.8),
            (r"\bfeel inferior\b", 1.5),
            (r"\bcan't measure up\b", 1.2),
        ],
        "self-criticism": [  # DEFINING 'self-criticism' (with hyphen)
            (r"\bself[- ]critic(al|ism)\b", 2.0),
            (r"\bhard on myself\b", 1.8),
            (r"\bbeat myself up\b", 1.8),
            (r"\btoo judgmental of myself\b", 1.5),
            (r"\bmy own worst critic\b", 1.5),
            (r"\bshould have done better\b", 1.2),
        ],
    }

    @classmethod
    def get_keywords_for_theme(cls, theme: str) -> List[str]:
        """Get keywords for a specific theme."""
        return cls.THERAPEUTIC_THEMES.get(theme.lower(), {}).get("keywords", [])

    @classmethod
    def get_approach_for_theme(cls, theme: Optional[str]) -> Union[str, List[str]]:
        if not theme:
            logger.warning("get_approach_for_theme called with an empty or None theme. Returning default.")
            return "supportive_listening"

        theme_lower = theme.lower().strip()
        logger.debug(f"get_approach_for_theme: Received theme='{theme}', processed to='{theme_lower}'")

        theme_data = cls.THERAPEUTIC_THEMES.get(theme_lower, {})
        if not theme_data:
            logger.warning(f"Theme '{theme_lower}' not found in THERAPEUTIC_THEMES. Defaulting approach.")
            # Fallback: Check if the input theme is directly an approach itself
            if theme_lower in cls.APPROACH_TO_TEMPLATE or any(
                theme_lower == key.lower() for key in cls.APPROACH_TO_TEMPLATE
            ):
                logger.debug(f"Theme '{theme_lower}' matches a direct approach. Returning it.")
                return theme_lower
            return "supportive_listening"

        approaches = theme_data.get("approaches")
        if approaches and isinstance(approaches, list) and len(approaches) > 0:
            # Ensure all items in the list are strings and lowercased
            valid_approaches = [str(app).lower() for app in approaches if isinstance(app, str)]
            if valid_approaches:
                logger.debug(f"For theme '{theme_lower}', found approaches: {valid_approaches}")
                return valid_approaches
            logger.warning(
                f"Theme '{theme_lower}' has an empty or invalid 'approaches' list after filtering. Defaulting."
            )
            return "supportive_listening"
        elif isinstance(approaches, str):  # If 'approaches' is a single string
            logger.debug(f"For theme '{theme_lower}', found single string approach: {approaches.lower()}")
            return approaches.lower()
        else:
            logger.warning(
                f"No valid 'approaches' defined for theme '{theme_lower}' or format is incorrect. Defaulting. Approaches found: {approaches}"
            )
            return "supportive_listening"

    @classmethod
    def get_template_for_approach(cls, approach_or_approaches: Optional[Union[str, List[str]]]) -> Optional[str]:
        """
        Retrieves the template name for a given therapeutic approach or list of approaches
        using the class's APPROACH_TO_TEMPLATE.
        If a list is provided, it returns the template for the first valid approach found.
        Returns None if no template is found.
        """
        if not approach_or_approaches:
            return None

        approaches_to_check: List[str] = []
        if isinstance(approach_or_approaches, str):
            approaches_to_check.append(approach_or_approaches)
        elif isinstance(approach_or_approaches, list):
            for item in approach_or_approaches:
                if isinstance(item, str):
                    approaches_to_check.append(item)

        for approach_str in approaches_to_check:
            normalized_approach = approach_str.lower().strip().replace("  ", " ")
            template = cls.APPROACH_TO_TEMPLATE.get(normalized_approach)
            if template:
                return template

            normalized_clean = "".join(c for c in normalized_approach if c.isalnum())
            if not normalized_clean:
                continue
            for key, value in cls.APPROACH_TO_TEMPLATE.items():
                key_clean = "".join(c for c in key if c.isalnum())
                if key_clean == normalized_clean:
                    return value

        return None

    @classmethod
    def get_template_for_theme(cls, theme: str) -> str:
        """
        Get template for a theme/topic by first getting its approach,
        then mapping approach to template. Defaults to 'empathy_validation'
        if no specific template is found.
        """
        approach = cls.get_approach_for_theme(theme)  # This can be str or List[str]
        template_name = cls.get_template_for_approach(approach)  # This is Optional[str]

        if template_name:
            return template_name
        logger.debug(
            "No specific template found for theme '%s' (via approach '%s'). Defaulting to 'empathy_validation'.",
            theme,
            approach,
        )
        return "empathy_validation"  # Provide a default string

    @classmethod
    def get_human_readable_name(cls, theme: str) -> str:
        """Get human-readable name for a theme."""
        return cls.THERAPEUTIC_THEMES.get(theme.lower(), {}).get("human_readable", "supportive_listening")

    @classmethod
    def get_keywords_for_taxonomy(cls, category: str) -> List[str]:
        """Get keywords for a specific taxonomy category (from enhanced taxonomy)."""
        return cls.ENHANCED_TAXONOMY.get(category.lower(), [])

    @classmethod
    def find_theme_for_keyword(cls, keyword: str) -> Optional[str]:
        """Find the appropriate theme for a given keyword using all available taxonomies."""
        if not keyword:
            return "supportive_listening"

        keyword = keyword.lower()

        # First check THERAPEUTIC_THEMES
        for theme, data in cls.THERAPEUTIC_THEMES.items():
            if any(k.lower() == keyword or keyword in k.lower() for k in data.get("keywords", [])):
                return theme

        # Then check ENHANCED_TAXONOMY
        for theme, keywords in cls.ENHANCED_TAXONOMY.items():
            if any(k.lower() == keyword or keyword in k.lower() for k in keywords):
                return theme

        return "supportive_listening"

    @classmethod
    def get_all_emotion_patterns(cls) -> Dict[str, List[Tuple[str, float]]]:
        """Get all emotion detection patterns."""
        return cls.EMOTION_PATTERNS

    @classmethod
    def get_emotion_patterns(cls) -> Dict[str, List[Tuple[str, float]]]:
        """Get only pure emotion patterns for emotion detection."""
        return cls.CORE_EMOTIONS

    @classmethod
    def get_clinical_patterns(cls) -> Dict[str, List[Tuple[str, float]]]:
        """Get clinical patterns for mental health condition detection."""
        return cls.CLINICAL_PATTERNS

    @classmethod
    def get_interpersonal_patterns(cls) -> Dict[str, List[Tuple[str, float]]]:
        """Get interpersonal patterns for relationship/identity detection."""
        return cls.INTERPERSONAL_PATTERNS

    @classmethod
    def get_all_patterns(cls) -> Dict[str, List[Tuple[str, float]]]:
        """Get all patterns combined (for backward compatibility)."""
        all_patterns = {}
        all_patterns.update(cls.CORE_EMOTIONS)
        all_patterns.update(cls.CLINICAL_PATTERNS)
        all_patterns.update(cls.INTERPERSONAL_PATTERNS)
        return all_patterns

    @classmethod
    def detect_theme_from_text(cls, text: str, use_patterns: bool = True) -> str:
        """
        Enhanced theme detection using both keywords and emotion patterns.

        Args:
            text: The input text to analyze
            use_patterns: Whether to also use emotion patterns for detection

        Returns:
            str: The detected theme name
        """
        if not text or not text.strip():
            return "general_support"

        text_lower = text.lower()

        # 1. Keyword-based detection (same as basic version)
        theme_scores = cls._score_themes_by_keywords(text_lower)

        # 2. Pattern-based detection (if enabled)
        if use_patterns:
            pattern_scores = cls._score_themes_by_patterns(text_lower)

            # Combine scores
            for theme, score in pattern_scores.items():
                if theme in theme_scores:
                    theme_scores[theme] += score * 0.5  # Weight pattern matches lower
                else:
                    theme_scores[theme] = score * 0.5

        # Return best match
        if theme_scores:
            best_theme = max(theme_scores.items(), key=lambda x: x[1])[0]
            logger.debug(f"detect_theme_from_text: Best theme '{best_theme}' with score {theme_scores[best_theme]}")
            return best_theme

        return "general_support"

    @classmethod
    def _score_themes_by_keywords(cls, text_lower: str) -> Dict[str, float]:
        """Score themes based on keyword matches."""
        theme_scores: Dict[str, float] = {}

        # Check THERAPEUTIC_THEMES
        for theme, theme_data in cls.THERAPEUTIC_THEMES.items():
            keywords = theme_data.get("keywords", [])
            score = 0

            for keyword in keywords:
                keyword_lower = keyword.lower()
                if keyword_lower in text_lower:
                    score += 2

                if re.search(r"\b" + re.escape(keyword_lower) + r"\b", text_lower):
                    score += 1

            if score > 0:
                theme_scores[theme] = score

        return theme_scores

    @classmethod
    def _score_themes_by_patterns(cls, text_lower: str) -> Dict[str, float]:
        """Score themes based on emotion pattern matches using existing TherapeuticMappings."""
        theme_scores: Dict[str, float] = {}  # Add explicit type annotation

        for emotion, patterns in cls.CORE_EMOTIONS.items():
            total_score = 0.0  # Make sure it's float
            for pattern, weight in patterns:
                matches = re.findall(pattern, text_lower)
                if matches:
                    total_score += len(matches) * weight

            if total_score > 0:
                # Find which themes contain this emotion in their emotions list
                matching_themes = []
                for theme, theme_data in cls.THERAPEUTIC_THEMES.items():
                    emotions_list = theme_data.get("emotions", [])
                    if emotion in emotions_list:
                        matching_themes.append(theme)

                # If no direct match found, map the emotion to themes
                if not matching_themes:
                    matching_themes = ["general_support"]  # fallback

                # Distribute score across matching themes
                for theme in matching_themes:
                    score_per_theme = total_score / len(matching_themes)
                    if theme in theme_scores:
                        theme_scores[theme] += score_per_theme
                    else:
                        theme_scores[theme] = score_per_theme

        return theme_scores
