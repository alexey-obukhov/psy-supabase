# Psy-Supabase

[![PsySupabase CI](https://github.com/alexey-obukhov/psy-supabase/actions/workflows/psy_supabase_ci.yml/badge.svg)](https://github.com/alexey-obukhov/psy-supabase/actions/workflows/psy_supabase_ci.yml)
[![Python 3.8 | 3.10](https://img.shields.io/badge/python-3.8%20%7C%203.10-blue)](https://www.python.org/downloads/)

## Code Quality Metrics

| Python Version | PyLint Score | Test Coverage |
|----------------|--------------|---------------|
| Python 3.8 | [![PyLint 3.8](https://alexey-obukhov.github.io/psy-supabase/badges/pylint-py3.8.svg)](https://github.com/alexey-obukhov/psy-supabase/blob/psy-supabase-pkg/reports/pylint-report.txt) | [![Coverage 3.8](https://alexey-obukhov.github.io/psy-supabase/badges/coverage-py3.8.svg)](https://github.com/alexey-obukhov/psy-supabase/blob/psy-supabase-pkg/coverage.xml) |
| Python 3.10 | [![PyLint 3.10](https://alexey-obukhov.github.io/psy-supabase/badges/pylint-py3.10.svg)](https://github.com/alexey-obukhov/psy-supabase/blob/psy-supabase-pkg/reports/pylint-report.txt) | [![Coverage 3.10](https://alexey-obukhov.github.io/psy-supabase/badges/coverage-py3.10.svg)](https://github.com/alexey-obukhov/psy-supabase/blob/psy-supabase-pkg/coverage.xml) |

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A psychological AI backend using Supabase for knowledge storage and retrieval.

# Pain Point Detection in AI Therapeutic Conversations

By leveraging pgvector's capabilities, there was created a more psychologically-informed AI assistant that can identify recurring themes and potential areas of psychological distress.

## Recent Enhancements

This innovation represents a significant step forward in therapeutic chatbots because:

1. **Pattern Recognition**: Rather than treating each user message in isolation, I'm using vector similarity to build a psychological profile over time

2. **Depth-oriented Responses**: The system now adapts its therapeutic approach based on detected pain points, similar to how a real therapist would follow threads of significance

3. **User-centered Experience**: By remembering and recognizing when users circle back to important topics, the conversation feels more coherent and thoughtful

4. **Efficient Vector Processing**: By performing most similarity operations directly in the database with pgvector, I maintain good performance while gaining rich psychological insights

5. **Dynamic Topic Detection**: Using natural language processing to identify psychological topics in user messages and retrieve relevant knowledge from my database

6. **Token-Aware Text Generation**: Implementing token-based prompt management to ensure optimal utilization of model context windows and prevent truncation issues

7. **Template Optimization**: Creating concise templates that maintain therapeutic quality while fitting within model constraints

8. **Fallback Mechanisms**: Implementing multi-level fallbacks to ensure reliable responses even when primary generation methods encounter issues

## Technical Improvements

My technical infrastructure has been enhanced with:

1. **Vector Search needs to be adjusted**: Truncate each sentence into valuable pieces before indexing or querying. This improves retrieval accuracy and ensures the most relevant information is represented in the vector database.

2. **Custom Logger Integration [prismalog](https://pypi.org/project/prismalog/)**: Replaced standard logging with own colorful, more readable, fast prismalog system

3. **Memory Optimization**: Added GPU memory management to gracefully handle CUDA out-of-memory scenarios with automatic CPU fallback

4. **Distributed Knowledge Management**: Implemented a system that can dynamically query the knowledge base mid-conversation when specific psychological topics arise

5. **Token-Based Prompt Management**: Added intelligent token counting and optimization to ensure prompts never exceed model context limits

6. **Smart Truncation**: Implemented priority-based truncation that preserves system instructions and user questions while reducing less essential content

7. **Token Monitoring**: Added detailed logging of token counts to track model utilization and identify optimization opportunities

8. **Add Full Customization** Integrate full customization via separate program and logging configuration files to tailor user interactions and system behavior based on individual needs and preferences.

## Next Steps

Planned Enhancements

1. **Emotion Trajectory**: Track a client’s emotional changes over time.

2. **AI Integration**: Integrate AI into the project as a 'knowledge database' to enhance response generation from user input on life, problems, and therapy expectations.

## Implementation Status

- ✅ Basic pain point detection architecture
- ✅ Vector similarity clustering
- ✅ Dynamic template selection
- ✅ Psychological topic extraction
- ✅ Memory-efficient processing
- ✅ Custom logging integration
- ✅ Add Full Customization
- ✅ Token-aware text generation
- ✅ Smart prompt truncation
- ✅ Enhanced Prompt Manager
- ✅ Token-efficient templates
- ✅ Enhancing Vector Search by Segmenting Text into Meaningful Units
- 📅 Integrating Emotional trajectory tracking (planned)
- 📅 AI Integration (planned)

By continuing to refine these capabilities, I'm building an AI therapeutic assistant that provides increasingly personalized, psychologically-informed support while maintaining operational efficiency. Looking forward to collaborate with interested people.

# psy-supabase

The code I have written for this project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

Please refer to manual how to get required free supabase anon key and project url required by this program [Supabase Setup Manual](https://alexey-obukhov.github.io/supabase_login.html)

*Beaware, these project uses libraries which may have their own respective licenses.
*Disclaimer: This AI tool is **not** a replacement for professional therapy and should **not** be used as a substitute for qualified mental health care.
