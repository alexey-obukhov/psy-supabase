from psy_supabase.utilities.therapeutic_mappings import _detect_themes

def test_anxiety_detection():
    questions = ["I feel anxious about my job.", "I worry a lot."]
    expected = {"anxiety"}
    assert _detect_themes(questions) == expected

def test_depression_detection():
    questions = ["I feel so sad and unmotivated.", "I am tired all the time."]
    expected = {"depression"}
    assert _detect_themes(questions) == expected

def test_trauma_detection():
    questions = ["I have flashbacks from the traumatic event."]
    expected = {"trauma"}
    assert _detect_themes(questions) == expected

def test_relationship_detection():
    questions = ["My partner and I are having issues."]
    expected = {"relationship"}
    assert _detect_themes(questions) == expected

def test_family_detection():
    questions = ["I have a great relationship with my mother."]
    expected = {"family"}
    assert _detect_themes(questions) == expected

def test_work_detection():
    questions = ["I am stressed about my job."]
    expected = {"work"}
    assert _detect_themes(questions) == expected

def test_self_esteem_detection():
    questions = ["I feel inadequate and not good enough."]
    expected = {"self-esteem"}
    assert _detect_themes(questions) == expected

def test_identity_detection():
    questions = ["I am trying to find out who I am."]
    expected = {"identity"}
    assert _detect_themes(questions) == expected

def test_grief_detection():
    questions = ["I am struggling with the loss of a loved one."]
    expected = {"grief"}
    assert _detect_themes(questions) == expected

def test_addiction_detection():
    questions = ["I have been trying to quit smoking."]
    expected = {"addiction"}
    assert _detect_themes(questions) == expected

def test_anger_detection():
    questions = ["I feel angry and frustrated."]
    expected = {"anger"}
    assert _detect_themes(questions) == expected

def test_trust_detection():
    questions = ["I feel betrayed by my friend."]
    expected = {"trust"}
    assert _detect_themes(questions) == expected

def test_guilt_detection():
    questions = ["I feel so much guilt about my actions."]
    expected = {"guilt"}
    assert _detect_themes(questions) == expected

def test_criticism_detection():
    questions = ["I feel criticized by my peers."]
    expected = {"criticism"}
    assert _detect_themes(questions) == expected

def test_rejection_detection():
    questions = ["I feel rejected by my peers."]
    expected = {"rejection"}
    assert _detect_themes(questions) == expected

def test_inadequacy_detection():
    questions = ["I often feel inadequate."]
    expected = {"inadequacy"}
    assert _detect_themes(questions) == expected

def test_jealousy_detection():
    questions = ["I feel jealous of my friend's success."]
    expected = {"jealousy"}
    assert _detect_themes(questions) == expected

def test_loneliness_detection():
    questions = ["I feel so lonely and isolated."]
    expected = {"loneliness"}
    assert _detect_themes(questions) == expected

def test_insecurity_detection():
    questions = ["I often feel insecure about my abilities."]
    expected = {"insecurity"}
    assert _detect_themes(questions) == expected