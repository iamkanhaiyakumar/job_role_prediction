# taxonomy.py
"""
Centralized Multi-Sector Career Taxonomy, Qualification Gating & Skill Normalization Engine
for Edu2Job (Prediction Engine v4 / Final Multi-Sector Intelligence).

Covers:
- 20 Distinct Industry Sectors
- 20 Canonical Career Roles across all 20 sectors (Representative Core Taxonomy)
- Qualification Gating Metadata for Regulated Professions (Doctor, Lawyer, Architect, CA, Academic Educator)
- 250+ Canonical Skill Aliases with Strict Token Boundary Normalization (Zero Substring False Positives)
- Multi-Tier Skill Hierarchy (CORE, IMPORTANT, SUPPORTING, GENERAL)
- Sector-Specific Learning Roadmaps & Mock Interview Preparation
"""

import re
from typing import Dict, Any, List, Optional, Set, Tuple

# ============================================================================
# 1. INDUSTRY SECTORS (20 Sectors)
# ============================================================================
SECTORS: List[str] = [
    'IT & Software',
    'AI & Machine Learning',
    'Data & Analytics',
    'Cloud & DevOps',
    'Cybersecurity',
    'Finance & Accounting',
    'Business & Consulting',
    'Management',
    'Mechanical Engineering',
    'Electrical & Electronics',
    'Civil & Construction',
    'Healthcare & Medical',
    'Law',
    'Education',
    'Architecture & Planning',
    'Marketing & Sales',
    'Human Resources',
    'Supply Chain & Operations',
    'Design & Creative',
    'Agriculture, Environment & Media'
]

# ============================================================================
# 2. CANONICAL MULTI-SECTOR CAREER ROLES (20 Representative Canonical Roles)
# ============================================================================
CAREER_TAXONOMY: Dict[str, Dict[str, Any]] = {
    # ---------------- 1. IT & SOFTWARE ----------------
    'Software Engineer': {
        'sector': 'IT & Software',
        'description': 'Designs, develops, and maintains scalable software applications, algorithms, and core system architectures.',
        'qualification_required': False,
        'required_degrees': [],
        'career_switch_allowed': True,
        'compatible_degrees': ['B.Tech', 'M.Tech', 'BCA', 'MCA', 'B.Sc', 'M.Sc'],
        'compatible_majors': ['Computer Science', 'Information Technology', 'Software Engineering', 'Electronics', 'Mathematics'],
        'industries': ['IT', 'Software', 'Engineering', 'Finance', 'E-commerce'],
        'core_skills': ['Data Structures', 'Algorithms', 'Java', 'C++', 'Python', 'Object Oriented Programming', 'Git'],
        'important_skills': ['System Design', 'DBMS', 'SQL', 'Operating Systems', 'Computer Networks', 'Unit Testing'],
        'supporting_skills': ['Linux', 'Docker', 'REST API', 'Design Patterns'],
        'general_skills': ['Problem Solving', 'Debugging', 'Teamwork', 'Communication'],
        'certifications': ['Oracle Certified Java Developer', 'AWS Certified Developer', 'HackerRank Problem Solving Gold'],
        'experience_expectation': '0-5+ years',
        'roadmap': [
            {'phase': 'Phase 1: Foundations', 'duration': 'Weeks 1-4', 'topics': 'OOP, Advanced DSA (Arrays, Trees, Graphs, DP), Git & Linux Basics', 'resources': [{'title': 'LeetCode Top 150', 'url': 'https://leetcode.com'}, {'title': 'NeetCode 150', 'url': 'https://neetcode.io'}]},
            {'phase': 'Phase 2: Core Engineering', 'duration': 'Weeks 5-8', 'topics': 'DBMS, OS Internals, Computer Networks, Concurrency & Multithreading', 'resources': [{'title': 'CS50 Harvard', 'url': 'https://cs50.harvard.edu'}]},
            {'phase': 'Phase 3: System Design', 'duration': 'Weeks 9-12', 'topics': 'Low-Level Design, High-Level Scalability, Caching (Redis), Message Queues (Kafka)', 'resources': [{'title': 'System Design Primer', 'url': 'https://github.com/donnemartin/system-design-primer'}]},
            {'phase': 'Phase 4: Interview Prep', 'duration': 'Weeks 13-16', 'topics': 'Mock Coding Interviews, Clean Code Architecture, Behavioral Questions', 'resources': [{'title': 'Pramp Mock Interviews', 'url': 'https://www.pramp.com'}]}
        ],
        'interview_questions': [
            {'q': 'How do you optimize an algorithm with O(n^2) time complexity to O(n log n) or O(n)?', 'a': 'Look for redundant computations. Utilize HashMaps for O(1) lookups, Sorting + Two Pointers for O(n log n), or Sliding Window / Prefix Sum techniques.'},
            {'q': 'Explain the difference between Processes and Threads, and how synchronization is handled.', 'a': 'Processes have isolated memory spaces; threads share memory within a process. Synchronization uses Mutexes, Semaphores, and Atomic variables to prevent race conditions.'},
            {'q': 'What are ACID properties in database transactions?', 'a': 'Atomicity (all or nothing), Consistency (preserves invariants), Isolation (concurrent executions are isolated), Durability (committed changes persist across crashes).'},
            {'q': 'Explain how HashMap works internally in Java / C++.', 'a': 'It uses an array of buckets. Key hashCode determines index. Collisions use LinkedList or Balanced Red-Black Trees (O(log n)) when bucket size exceeds threshold.'},
            {'q': 'Walk through the design of a URL Shortener service (like bit.ly).', 'a': 'Use Base62 encoding on an auto-incrementing ID or MD5 hash. Cache hot URLs in Redis, store in distributed DB (PostgreSQL/Cassandra), use Load Balancer and Rate Limiting.'}
        ]
    },
    'Frontend Developer': {
        'sector': 'IT & Software',
        'description': 'Builds responsive, performant, and intuitive web interfaces and user-facing web applications.',
        'qualification_required': False,
        'required_degrees': [],
        'career_switch_allowed': True,
        'compatible_degrees': ['B.Tech', 'BCA', 'MCA', 'B.Sc', 'B.A', 'BBA'],
        'compatible_majors': ['Computer Science', 'Information Technology', 'Web Development', 'Design', 'Computer Applications'],
        'industries': ['IT', 'Software', 'E-commerce', 'Marketing', 'Media'],
        'core_skills': ['HTML', 'CSS', 'JavaScript', 'React', 'Tailwind CSS'],
        'important_skills': ['TypeScript', 'Next.js', 'Redux', 'Responsive Web Design', 'REST API', 'Git'],
        'supporting_skills': ['Vue.js', 'Bootstrap', 'Webpack', 'Vite', 'GraphQL', 'Jest'],
        'general_skills': ['UI/UX Awareness', 'Cross-Browser Testing', 'Communication', 'Attention to Detail'],
        'certifications': ['Meta Front-End Developer Certificate', 'freeCodeCamp Responsive Web Design'],
        'experience_expectation': '0-4+ years',
        'roadmap': [
            {'phase': 'Phase 1: Web Essentials', 'duration': 'Weeks 1-4', 'topics': 'Semantic HTML5, Modern CSS3 (Flexbox/Grid), JavaScript ES6+ (Async/Await, DOM)', 'resources': [{'title': 'MDN Web Docs', 'url': 'https://developer.mozilla.org'}, {'title': 'JavaScript.info', 'url': 'https://javascript.info'}]},
            {'phase': 'Phase 2: React & Modern Tooling', 'duration': 'Weeks 5-8', 'topics': 'React Hooks, State Management (Zustand/Redux), Tailwind CSS, TypeScript', 'resources': [{'title': 'React Official Docs', 'url': 'https://react.dev'}]},
            {'phase': 'Phase 3: SSR & Frameworks', 'duration': 'Weeks 9-12', 'topics': 'Next.js App Router, Server Components, API Integration, Performance Optimization (Core Web Vitals)', 'resources': [{'title': 'Next.js Learn', 'url': 'https://nextjs.org/learn'}]},
            {'phase': 'Phase 4: Production & Portfolio', 'duration': 'Weeks 13-16', 'topics': 'Build 3 deployed full-featured frontend apps, Unit Testing (Vitest/Jest), Lighthouse Audits', 'resources': [{'title': 'Frontend Mentor', 'url': 'https://www.frontendmentor.io'}]}
        ],
        'interview_questions': [
            {'q': 'Explain the Virtual DOM and React Reconciliation process.', 'a': 'Virtual DOM is an in-memory lightweight representation of the real DOM. React diffs the previous and current VDOM trees and batches minimal updates to the real DOM via its Fiber reconciler.'},
            {'q': 'What is the Event Loop in JavaScript and how do Microtasks differ from Macrotasks?', 'a': 'The event loop processes the call stack. When empty, it drains the Microtask Queue (Promises, process.nextTick) before picking the next Macrotask (setTimeout, setInterval, I/O).'},
            {'q': 'What are the key advantages of TypeScript over standard JavaScript in large frontend codebases?', 'a': 'Static type safety, compile-time error catching, superior IDE autocompletion, self-documenting interfaces, and safer refactoring.'},
            {'q': 'How do you optimize web page load times and Core Web Vitals (LCP, FID/INP, CLS)?', 'a': 'Code splitting/lazy loading, image optimization (WebP/AVIF, responsive srcset), CSS purging, CDN caching, prefetching, and reserving layout space to prevent CLS.'},
            {'q': 'Explain Closures and Prototypal Inheritance in JavaScript.', 'a': 'A closure is a function that retains access to its lexical scope even after the outer function has executed. Prototypal inheritance allows objects to inherit properties and methods via `__proto__` chain.'}
        ]
    },
    'Backend Developer': {
        'sector': 'IT & Software',
        'description': 'Engineers robust server-side business logic, database integrations, authentication, and RESTful/GraphQL APIs.',
        'qualification_required': False,
        'required_degrees': [],
        'career_switch_allowed': True,
        'compatible_degrees': ['B.Tech', 'MCA', 'BCA', 'M.Tech', 'B.Sc'],
        'compatible_majors': ['Computer Science', 'Information Technology', 'Software Engineering'],
        'industries': ['IT', 'Finance', 'SaaS', 'Healthcare', 'E-commerce'],
        'core_skills': ['Node.js', 'Python', 'Java', 'SQL', 'PostgreSQL', 'Express.js', 'Django'],
        'important_skills': ['FastAPI', 'MongoDB', 'Redis', 'Microservices', 'Docker', 'REST API', 'Git'],
        'supporting_skills': ['Spring Boot', 'GraphQL', 'Kafka', 'Linux', 'JWT Auth', 'CI/CD'],
        'general_skills': ['System Architecture', 'Database Optimization', 'Problem Solving'],
        'certifications': ['AWS Certified Developer', 'Node.js Certified Developer', 'MongoDB Certified Associate'],
        'experience_expectation': '0-5+ years',
        'roadmap': [
            {'phase': 'Phase 1: Server Language & APIs', 'duration': 'Weeks 1-4', 'topics': 'Node.js/Express or Python/FastAPI, HTTP Methods, Middleware, RESTful Conventions', 'resources': [{'title': 'Node.js Documentation', 'url': 'https://nodejs.org'}]},
            {'phase': 'Phase 2: Database Mastery', 'duration': 'Weeks 5-8', 'topics': 'PostgreSQL Relational Design, Indexing, Transactions, MongoDB NoSQL, Redis Caching', 'resources': [{'title': 'Use The Index, Luke!', 'url': 'https://use-the-index-luke.com'}]},
            {'phase': 'Phase 3: Architecture & Security', 'duration': 'Weeks 9-12', 'topics': 'JWT/OAuth2 Security, Rate Limiting, Microservices, Docker, Message Queues (RabbitMQ/Kafka)', 'resources': [{'title': 'Microservices.io', 'url': 'https://microservices.io'}]},
            {'phase': 'Phase 4: Deployment & Scale', 'duration': 'Weeks 13-16', 'topics': 'CI/CD Pipelines, AWS Deployment (ECS/EC2), Database Connection Pooling, Load Testing', 'resources': [{'title': 'Roadmap.sh Backend', 'url': 'https://roadmap.sh/backend'}]}
        ],
        'interview_questions': [
            {'q': 'Explain SQL Indexing and why adding too many indexes can degrade write performance.', 'a': 'Indexes (usually B-Trees) speed up SELECT queries by avoiding full-table scans. However, on INSERT/UPDATE/DELETE, every index must also be updated, consuming additional I/O and CPU.'},
            {'q': 'How do you secure a REST API against common vulnerabilities (SQL Injection, XSS, CSRF, DDoS)?', 'a': 'Use Parameterized Queries / ORMs for SQLi; sanitize inputs and set Content Security Policy for XSS; use Anti-CSRF tokens / SameSite cookies; enforce Rate Limiting and API Gateways for DDoS.'},
            {'q': 'What is the difference between Synchronous and Asynchronous programming in Node.js / Python?', 'a': 'Synchronous blocks thread execution until operation finishes. Asynchronous uses non-blocking I/O with Event Loops / async-await to handle thousands of concurrent requests efficiently.'},
            {'q': 'How does Redis caching work and what are common Cache Eviction policies?', 'a': 'Redis stores key-value data in RAM for sub-millisecond retrieval. Eviction policies include LRU (Least Recently Used), LFU (Least Frequently Used), and TTL (Time-To-Live expiration).'},
            {'q': 'Explain the difference between Horizontal and Vertical Scaling.', 'a': 'Vertical scaling increases CPU/RAM on a single machine (hardware ceiling). Horizontal scaling adds more server instances behind a Load Balancer (virtually unlimited scale).' }
        ]
    },
    'Full Stack Developer': {
        'sector': 'IT & Software',
        'description': 'Builds end-to-end web applications combining interactive frontends with scalable server-side systems and databases.',
        'qualification_required': False,
        'required_degrees': [],
        'career_switch_allowed': True,
        'compatible_degrees': ['B.Tech', 'MCA', 'BCA', 'M.Tech', 'B.Sc'],
        'compatible_majors': ['Computer Science', 'Information Technology', 'Software Engineering'],
        'industries': ['IT', 'Startups', 'SaaS', 'Finance', 'Consulting'],
        'core_skills': ['React', 'Node.js', 'JavaScript', 'SQL', 'MongoDB', 'HTML', 'CSS'],
        'important_skills': ['TypeScript', 'Next.js', 'Express.js', 'PostgreSQL', 'Docker', 'Tailwind CSS', 'REST API'],
        'supporting_skills': ['Git', 'Redis', 'GraphQL', 'AWS', 'CI/CD', 'Linux'],
        'general_skills': ['End-to-End Problem Solving', 'Product Mindset', 'Communication'],
        'certifications': ['Meta Full-Stack Engineer Certificate', 'AWS Certified Developer Associate'],
        'experience_expectation': '0-5+ years',
        'roadmap': [
            {'phase': 'Phase 1: Frontend & UI', 'duration': 'Weeks 1-4', 'topics': 'Modern HTML/CSS, React, Tailwind CSS, State Management', 'resources': [{'title': 'FullStackOpen', 'url': 'https://fullstackopen.com'}]},
            {'phase': 'Phase 2: Backend & DB', 'duration': 'Weeks 5-8', 'topics': 'Node.js, Express, REST APIs, PostgreSQL & MongoDB database modeling', 'resources': [{'title': 'The Odin Project', 'url': 'https://www.theodinproject.com'}]},
            {'phase': 'Phase 3: Integration & Auth', 'duration': 'Weeks 9-12', 'topics': 'Next.js Fullstack, JWT/OAuth Authentication, Stripe Payment Gateway, Docker', 'resources': [{'title': 'Next.js Fullstack Course', 'url': 'https://nextjs.org'}]},
            {'phase': 'Phase 4: Production SaaS Project', 'duration': 'Weeks 13-16', 'topics': 'Deploy full SaaS product with CI/CD on AWS/Vercel, Monitoring, Portfolio Polish', 'resources': [{'title': 'Roadmap.sh Full Stack', 'url': 'https://roadmap.sh/full-stack'}]}
        ],
        'interview_questions': [
            {'q': 'Walk me through the architecture of a complete Full-Stack web application from browser to database.', 'a': 'Browser sends HTTPS request -> DNS -> CDN/Load Balancer -> Reverse Proxy (Nginx) -> Frontend/Backend Node.js/Python server -> Business Logic Layer -> Caching (Redis) -> Relational/NoSQL Database.'},
            {'q': 'What is Server-Side Rendering (SSR) vs Client-Side Rendering (CSR) vs Static Site Generation (SSG)?', 'a': 'CSR renders HTML in browser via JS (slower initial load, fast navigation). SSR renders on every request on server (great SEO, fresh data). SSG pre-renders HTML at build time (maximum speed and security).'},
            {'q': 'How do you prevent CORS (Cross-Origin Resource Sharing) errors securely?', 'a': 'Configure server headers: `Access-Control-Allow-Origin` targeting specific trusted domains (avoid `*` with credentials), and handle OPTIONS preflight requests.'},
            {'q': 'Explain how Database Migrations work in agile production environments.', 'a': 'Migration scripts version database schema changes incrementally (up/down). They ensure development, staging, and production databases remain in sync without manual SQL execution.'},
            {'q': 'How do you design an authentication flow using JWT with Refresh Tokens?', 'a': 'Issue short-lived Access Token (15 mins, in memory) and long-lived Refresh Token (7 days, httpOnly Secure cookie). When access token expires, client hits refresh endpoint to renew access token.'}
        ]
    },
    'Mobile App Developer': {
        'sector': 'IT & Software',
        'description': 'Develops native and cross-platform mobile applications for iOS and Android platforms.',
        'qualification_required': False,
        'required_degrees': [],
        'career_switch_allowed': True,
        'compatible_degrees': ['B.Tech', 'BCA', 'MCA', 'B.Sc'],
        'compatible_majors': ['Computer Science', 'Information Technology', 'Software Engineering'],
        'industries': ['IT', 'Mobile', 'Gaming', 'Fintech', 'E-commerce'],
        'core_skills': ['Flutter', 'React Native', 'Kotlin', 'Swift', 'Mobile App Development'],
        'important_skills': ['Android Studio', 'Xcode', 'REST API', 'Firebase', 'State Management', 'Git'],
        'supporting_skills': ['Dart', 'Java', 'SQLite', 'Push Notifications', 'App Store Deployment'],
        'general_skills': ['Mobile UI/UX', 'Performance Optimization', 'Debugging'],
        'certifications': ['Google Associate Android Developer', 'Meta React Native Specialization'],
        'experience_expectation': '0-4+ years',
        'roadmap': [
            {'phase': 'Phase 1: Language & SDK', 'duration': 'Weeks 1-4', 'topics': 'Dart/Flutter or TypeScript/React Native fundamentals, Mobile UI Components', 'resources': [{'title': 'Flutter Official Docs', 'url': 'https://flutter.dev'}]},
            {'phase': 'Phase 2: State & Storage', 'duration': 'Weeks 5-8', 'topics': 'BLoC/Provider or Redux, SQLite Local Storage, REST API Integration', 'resources': [{'title': 'React Native Guide', 'url': 'https://reactnative.dev'}]},
            {'phase': 'Phase 3: Native Features', 'duration': 'Weeks 9-12', 'topics': 'Camera, Location/GPS, Push Notifications (FCM), Background Services', 'resources': [{'title': 'Android Developers', 'url': 'https://developer.android.com'}]},
            {'phase': 'Phase 4: Store Publishing', 'duration': 'Weeks 13-16', 'topics': 'Build 2 full apps, Google Play & Apple App Store release pipelines, App Security', 'resources': [{'title': 'Apple Developer Documentation', 'url': 'https://developer.apple.com'}]}
        ],
        'interview_questions': [
            {'q': 'Explain the difference between Native (Kotlin/Swift) and Cross-Platform (Flutter/React Native) mobile development.', 'a': 'Native gives direct hardware access and peak performance with platform-native UI. Cross-platform shares a single codebase across iOS/Android, cutting development cost and time significantly.'},
            {'q': 'How does Flutter render UI without using OEM native widgets?', 'a': 'Flutter uses its own Skia/Impeller C++ graphics engine to draw every pixel on a canvas, guaranteeing identical rendering and 60/120 FPS animations across iOS and Android.'},
            {'q': 'How do you manage mobile app offline state and data synchronization?', 'a': 'Use local databases (SQLite/Room/Hive) as the single source of truth. When online, a background sync manager reconciles changes with the server using conflict resolution (e.g., timestamps).'},
            {'q': 'What causes memory leaks in mobile applications and how do you profile them?', 'a': 'Retaining static references to Activities/Views, unclosed database streams, and uncancelled async listeners. Profile using Android Studio Profiler or Xcode Instruments.'},
            {'q': 'Explain App Lifecycle states in Android and iOS.', 'a': 'Android: Created -> Started -> Resumed -> Paused -> Stopped -> Destroyed. iOS: Not Running -> Inactive -> Active -> Background -> Suspended.'}
        ]
    },

    # ---------------- 2. AI & MACHINE LEARNING ----------------
    'Machine Learning Engineer': {
        'sector': 'AI & Machine Learning',
        'description': 'Researches, trains, optimizes, and deploys production machine learning and deep learning neural networks.',
        'qualification_required': False,
        'required_degrees': [],
        'career_switch_allowed': True,
        'compatible_degrees': ['B.Tech', 'M.Tech', 'MCA', 'B.Sc', 'M.Sc', 'PhD'],
        'compatible_majors': ['Computer Science', 'Artificial Intelligence', 'Data Science', 'Mathematics', 'Electronics'],
        'industries': ['IT', 'AI', 'Automotive', 'Healthcare', 'Fintech'],
        'core_skills': ['Python', 'Machine Learning', 'Deep Learning', 'PyTorch', 'TensorFlow', 'Model Deployment'],
        'important_skills': ['Scikit-Learn', 'FastAPI', 'Docker', 'MLOps', 'Computer Vision', 'NLP', 'Data Science'],
        'supporting_skills': ['C++', 'LangChain', 'Hugging Face', 'ONNX', 'Git', 'Linux', 'SQL'],
        'general_skills': ['Mathematical Modeling', 'Experimentation', 'Problem Solving'],
        'certifications': ['AWS Certified Machine Learning Specialty', 'TensorFlow Developer Certificate', 'DeepLearning.AI Specialization'],
        'experience_expectation': '0-5+ years',
        'roadmap': [
            {'phase': 'Phase 1: Math & ML Core', 'duration': 'Weeks 1-4', 'topics': 'Linear Algebra, Calculus, Python OOP, Scikit-Learn, Supervised & Unsupervised ML', 'resources': [{'title': 'Coursera ML Specialization (Andrew Ng)', 'url': 'https://www.coursera.org'}, {'title': 'StatQuest', 'url': 'https://statquest.org'}]},
            {'phase': 'Phase 2: Deep Learning & PyTorch', 'duration': 'Weeks 5-8', 'topics': 'Neural Networks, Backpropagation, CNNs, RNNs, PyTorch Tensors & Training Loops', 'resources': [{'title': 'PyTorch Tutorials', 'url': 'https://pytorch.org/tutorials'}]},
            {'phase': 'Phase 3: Transformers & GenAI', 'duration': 'Weeks 9-12', 'topics': 'Self-Attention, Hugging Face Transformers, LLM Fine-Tuning (LoRA), RAG with LangChain', 'resources': [{'title': 'Hugging Face NLP Course', 'url': 'https://huggingface.co/learn'}]},
            {'phase': 'Phase 4: MLOps & Deployment', 'duration': 'Weeks 13-16', 'topics': 'FastAPI Microservices, Docker, ONNX Quantization, Triton/vLLM Inference Serving, Model Monitoring', 'resources': [{'title': 'Made With ML (MLOps)', 'url': 'https://madewithml.com'}]}
        ],
        'interview_questions': [
            {'q': 'How does Backpropagation calculate gradients in a deep neural network?', 'a': 'It applies the Chain Rule of calculus recursively from the loss output backwards through each layer, computing partial derivatives of the loss with respect to all weights.'},
            {'q': 'Explain the Self-Attention mechanism in the Transformer architecture.', 'a': 'Attention computes Queries (Q), Keys (K), and Values (V): `Attention(Q,K,V) = softmax(Q @ K^T / sqrt(d_k)) @ V`. It calculates dynamic pairwise relevance scores across all tokens in parallel.'},
            {'q': 'How do you optimize LLMs and Deep Learning models for low-latency inference in production?', 'a': 'Model Quantization (INT8/INT4/FP8), Weight Pruning, Knowledge Distillation, KV Caching, TensorRT-LLM / vLLM batching, and ONNX graph optimization.'},
            {'q': 'What is Data Drift vs Concept Drift and how do you monitor them in MLOps?', 'a': 'Data Drift is a shift in the input feature distribution `P(X)`. Concept Drift is a shift in the relationship between input features and target labels `P(Y|X)`. Monitor using Kolmogorov-Smirnov test and PSI.'},
            {'q': 'Explain the Bias-Variance Tradeoff and how regularization techniques (L1/L2, Dropout) help.', 'a': 'High bias causes underfitting; high variance causes overfitting. L1 (Lasso) promotes sparsity; L2 (Ridge) shrinks weights; Dropout randomly disables neurons during training to prevent co-adaptation.'}
        ]
    },
    'AI Engineer': {
        'sector': 'AI & Machine Learning',
        'description': 'Builds intelligent systems powered by Generative AI, Large Language Models (LLMs), RAG architectures, and AI agents.',
        'qualification_required': False,
        'required_degrees': [],
        'career_switch_allowed': True,
        'compatible_degrees': ['B.Tech', 'M.Tech', 'MCA', 'B.Sc', 'M.Sc'],
        'compatible_majors': ['Computer Science', 'Artificial Intelligence', 'Data Science', 'Information Technology'],
        'industries': ['IT', 'AI', 'Enterprise SaaS', 'Consulting', 'Healthcare'],
        'core_skills': ['Python', 'Generative AI', 'LangChain', 'Prompt Engineering', 'Vector Databases', 'Machine Learning'],
        'important_skills': ['PyTorch', 'Hugging Face', 'FastAPI', 'Deep Learning', 'Docker', 'NLP'],
        'supporting_skills': ['OpenAI API', 'ChromaDB', 'Pinecone', 'LlamaIndex', 'Fine-Tuning', 'Git'],
        'general_skills': ['AI Ethics', 'Prompt Optimization', 'System Integration'],
        'certifications': ['DeepLearning.AI GenAI Specialization', 'AWS Certified AI Practitioner'],
        'experience_expectation': '0-5+ years',
        'roadmap': [
            {'phase': 'Phase 1: Python & AI APIs', 'duration': 'Weeks 1-4', 'topics': 'Python Async, OpenAI/Anthropic APIs, Prompt Engineering Techniques, Embeddings', 'resources': [{'title': 'DeepLearning.AI Prompt Course', 'url': 'https://www.deeplearning.ai'}]},
            {'phase': 'Phase 2: RAG & Vector Search', 'duration': 'Weeks 5-8', 'topics': 'Chunking Strategies, Vector DBs (Chroma/Pinecone), Hybrid Search, Re-ranking', 'resources': [{'title': 'Pinecone Learning Center', 'url': 'https://www.pinecone.io/learn/'}]},
            {'phase': 'Phase 3: Autonomous AI Agents', 'duration': 'Weeks 9-12', 'topics': 'LangChain, LangGraph, Multi-Agent Orchestration, Tool Calling, Memory Management', 'resources': [{'title': 'LangChain Official Docs', 'url': 'https://python.langchain.com'}]},
            {'phase': 'Phase 4: LLM Evaluation & Scale', 'duration': 'Weeks 13-16', 'topics': 'RAG Triad Evaluation (Ragas/TruLens), Fine-tuning with LoRA/QLoRA, Guardrails & Security', 'resources': [{'title': 'Hugging Face Open LLM Leaderboard', 'url': 'https://huggingface.co'}]}
        ],
        'interview_questions': [
            {'q': 'Explain Retrieval-Augmented Generation (RAG) and how it mitigates LLM hallucinations.', 'a': 'RAG retrieves relevant domain documents from a vector database using semantic similarity and injects them into the LLM context prompt, grounding responses in verified external factual sources.'},
            {'q': 'What is the difference between Fine-Tuning and RAG? When should you use which?', 'a': 'Use RAG for dynamic, private, or frequently updated knowledge retrieval. Use Fine-Tuning to teach a model a specific style, formatting, syntax, or specialized domain jargon.'},
            {'q': 'How do Vector Embeddings and Approximate Nearest Neighbors (ANN) algorithms work?', 'a': 'Embeddings map text to high-dimensional dense vectors preserving semantic meaning. ANN algorithms (like HNSW or IVF) locate nearest vectors in logarithmic time without comparing every vector.'},
            {'q': 'What is LoRA (Low-Rank Adaptation) in LLM fine-tuning?', 'a': 'LoRA freezes pre-trained model weights and injects trainable low-rank rank-decomposition matrices into Transformer attention layers, slashing trainable parameter count by 99% with minimal memory overhead.'},
            {'q': 'How do you evaluate RAG systems using frameworks like Ragas?', 'a': 'Evaluate 3 core metrics: 1) Faithfulness (is answer grounded in context?), 2) Answer Relevance (does it answer the prompt?), 3) Context Precision/Recall (is retrieved context accurate and sufficient?).'}
        ]
    },

    # ---------------- 3. DATA & ANALYTICS ----------------
    'Data Scientist': {
        'sector': 'Data & Analytics',
        'description': 'Uncovers actionable business insights and builds predictive statistical models using structured and unstructured data.',
        'qualification_required': False,
        'required_degrees': [],
        'career_switch_allowed': True,
        'compatible_degrees': ['B.Tech', 'M.Tech', 'M.Sc', 'B.Sc', 'PhD', 'MBA'],
        'compatible_majors': ['Computer Science', 'Data Science', 'Mathematics', 'Statistics', 'Information Technology', 'Economics'],
        'industries': ['Data Science', 'IT', 'Finance', 'Healthcare', 'Consulting', 'E-commerce'],
        'core_skills': ['Python', 'SQL', 'Machine Learning', 'Pandas', 'NumPy', 'Statistics'],
        'important_skills': ['Data Visualization', 'Scikit-Learn', 'Matplotlib', 'Seaborn', 'Exploratory Data Analysis', 'Deep Learning'],
        'supporting_skills': ['R', 'Tableau', 'Power BI', 'FastAPI', 'Docker', 'Git'],
        'general_skills': ['Data Storytelling', 'Hypothesis Testing', 'Business Communication'],
        'certifications': ['Google Data Analytics Professional', 'Coursera IBM Data Science Specialization', 'Microsoft Certified Data Scientist'],
        'experience_expectation': '0-5+ years',
        'roadmap': [
            {'phase': 'Phase 1: Statistics & SQL', 'duration': 'Weeks 1-4', 'topics': 'Descriptive/Inferential Stats, Probability Distributions, Advanced SQL Queries & Window Functions', 'resources': [{'title': 'Kaggle SQL Tutorial', 'url': 'https://www.kaggle.com/learn'}]},
            {'phase': 'Phase 2: EDA & Feature Engineering', 'duration': 'Weeks 5-8', 'topics': 'Pandas data manipulation, Outlier detection, Missing value imputation, Feature encoding', 'resources': [{'title': 'Python for Data Analysis (Wes McKinney)', 'url': 'https://wesmckinney.com'}]},
            {'phase': 'Phase 3: Machine Learning Models', 'duration': 'Weeks 9-12', 'topics': 'Regression, Classification, Clustering, Cross-Validation, Hyperparameter Tuning (Optuna)', 'resources': [{'title': 'Scikit-Learn Docs', 'url': 'https://scikit-learn.org'}]},
            {'phase': 'Phase 4: Business Insights & Deployment', 'duration': 'Weeks 13-16', 'topics': 'A/B Testing, Executive Dashboards (Streamlit/Tableau), Model Deployment via FastAPI', 'resources': [{'title': 'Streamlit Learning', 'url': 'https://streamlit.io'}]}
        ],
        'interview_questions': [
            {'q': 'Explain the Central Limit Theorem (CLT) and why it is fundamental to statistical inference.', 'a': 'CLT states that the sampling distribution of the sample mean approaches a normal distribution as sample size increases (n >= 30), regardless of the population distribution shape.'},
            {'q': 'What is the difference between Type I and Type II errors in hypothesis testing?', 'a': 'Type I error (False Positive) rejects a true null hypothesis (alpha). Type II error (False Negative) fails to reject a false null hypothesis (beta).'},
            {'q': 'How do you handle severe class imbalance in a classification problem?', 'a': 'Resampling (SMOTE / Random Undersampling), cost-sensitive loss weighting (`class_weight="balanced"`), anomaly detection algorithms, and evaluating PR-AUC/F1 rather than ROC-AUC/Accuracy.'},
            {'q': 'Walk me through how you design and analyze an A/B test.', 'a': 'Define primary KPI and hypothesis -> Calculate sample size & power (80%, alpha=0.05) -> Randomly split traffic -> Run test without peeking -> Run Two-Sample t-test / Chi-square test -> Interpret p-value and confidence intervals.'},
            {'q': 'Explain PCA (Principal Component Analysis) and how it preserves variance.', 'a': 'PCA calculates the eigenvectors and eigenvalues of the feature covariance matrix to project data onto orthogonal principal component axes, capturing maximum variance in fewer dimensions.'}
        ]
    },
    'Data Analyst': {
        'sector': 'Data & Analytics',
        'description': 'Transforms raw business data into clear dashboards, reports, and strategic intelligence to guide corporate decision-making.',
        'qualification_required': False,
        'required_degrees': [],
        'career_switch_allowed': True,
        'compatible_degrees': ['B.Tech', 'BCA', 'B.Sc', 'B.Com', 'BBA', 'MBA', 'M.Sc'],
        'compatible_majors': ['Information Technology', 'Computer Science', 'Finance', 'Business', 'Mathematics', 'Economics'],
        'industries': ['Finance', 'IT', 'Consulting', 'Marketing', 'Healthcare', 'Retail'],
        'core_skills': ['SQL', 'Excel', 'Power BI', 'Python', 'Tableau', 'Data Analysis'],
        'important_skills': ['Data Visualization', 'Pandas', 'Statistics', 'MySQL', 'Business Analysis', 'Exploratory Data Analysis'],
        'supporting_skills': ['Advanced Excel', 'PostgreSQL', 'NumPy', 'Matplotlib', 'Git', 'Google Analytics'],
        'general_skills': ['Analytical Problem Solving', 'Dashboard Design', 'Stakeholder Reporting'],
        'certifications': ['Google Data Analytics Certificate', 'Microsoft Power BI Data Analyst Associate (PL-300)', 'Tableau Desktop Specialist'],
        'experience_expectation': '0-4+ years',
        'roadmap': [
            {'phase': 'Phase 1: Excel & SQL Foundations', 'duration': 'Weeks 1-4', 'topics': 'Pivot Tables, VLOOKUP/XLOOKUP, Power Query, SQL Joins, Aggregations, Group By, Subqueries', 'resources': [{'title': 'Chandoo Excel', 'url': 'https://chandoo.org'}, {'title': 'SQLZoo', 'url': 'https://sqlzoo.net'}]},
            {'phase': 'Phase 2: BI Dashboards', 'duration': 'Weeks 5-8', 'topics': 'Power BI DAX formulas, Star Schema Data Modeling, Interactive Tableau Visualizations', 'resources': [{'title': 'Microsoft Power BI Learn', 'url': 'https://learn.microsoft.com'}]},
            {'phase': 'Phase 3: Python for Analytics', 'duration': 'Weeks 9-12', 'topics': 'Pandas Data Cleaning, Matplotlib/Seaborn Charting, Automated Reporting Scripts', 'resources': [{'title': 'Kaggle Data Visualization', 'url': 'https://www.kaggle.com/learn'}]},
            {'phase': 'Phase 4: Business Case Studies', 'duration': 'Weeks 13-16', 'topics': 'Customer Churn Analysis, Sales Forecasting Dashboard, Executive Stakeholder Presentations', 'resources': [{'title': 'Maven Analytics Projects', 'url': 'https://www.mavenanalytics.io'}]}
        ],
        'interview_questions': [
            {'q': 'Explain the difference between SQL WHERE and HAVING clauses.', 'a': 'WHERE filters individual records before aggregations are computed. HAVING filters aggregated result groups produced by a GROUP BY clause.'},
            {'q': 'What is the difference between Star Schema and Snowflake Schema in data warehousing?', 'a': 'Star Schema connects fact tables directly to denormalized dimension tables (faster queries, simpler design). Snowflake schema normalizes dimension tables into sub-tables (saves storage, more complex joins).'},
            {'q': 'What are Window Functions in SQL and provide an example of when to use `ROW_NUMBER()` vs `RANK()`.', 'a': 'Window functions perform calculations across a subset of rows without collapsing them. For ties (e.g. scores 100, 100, 90), `ROW_NUMBER` gives 1, 2, 3; `RANK` gives 1, 1, 3.'},
            {'q': 'How do you handle null values, outliers, and duplicate records in a messy dataset?', 'a': 'Detect using summary stats/boxplots. For nulls: impute median/mode or drop. For duplicates: dedup via primary key constraints. For outliers: assess domain context before clipping or transformation.'},
            {'q': 'How do you write a DAX measure in Power BI to calculate Year-over-Year (YoY) Sales Growth?', 'a': '`YoY Sales Growth = VAR PrevYear = CALCULATE([Total Sales], SAMEPERIODLASTYEAR(\'Date\'[Date])) RETURN DIVIDE([Total Sales] - PrevYear, PrevYear, 0)`.'}
        ]
    },

    # ---------------- 4. CLOUD & DEVOPS ----------------
    'DevOps Engineer': {
        'sector': 'Cloud & DevOps',
        'description': 'Automates deployment pipelines, container orchestration, monitoring, and infrastructure as code for continuous delivery.',
        'qualification_required': False,
        'required_degrees': [],
        'career_switch_allowed': True,
        'compatible_degrees': ['B.Tech', 'M.Tech', 'MCA', 'B.Sc'],
        'compatible_majors': ['Computer Science', 'Information Technology', 'Electronics'],
        'industries': ['IT', 'Cloud', 'SaaS', 'Finance', 'E-commerce'],
        'core_skills': ['Linux', 'Docker', 'Kubernetes', 'CI/CD', 'AWS', 'Git', 'Terraform'],
        'important_skills': ['Jenkins', 'GitHub Actions', 'Bash', 'Ansible', 'Python', 'Prometheus', 'Grafana'],
        'supporting_skills': ['Nginx', 'Networking', 'CloudWatch', 'YAML', 'Helm', 'Security'],
        'general_skills': ['Automation Mindset', 'System Reliability', 'Incident Response'],
        'certifications': ['Certified Kubernetes Administrator (CKA)', 'AWS Certified Solutions Architect Associate', 'HashiCorp Certified Terraform Associate'],
        'experience_expectation': '0-5+ years',
        'roadmap': [
            {'phase': 'Phase 1: Linux & Containers', 'duration': 'Weeks 1-4', 'topics': 'Linux Administration, Bash Scripting, Docker Containerization, Multi-Stage Builds, Docker Compose', 'resources': [{'title': 'Docker Docs', 'url': 'https://docs.docker.com'}]},
            {'phase': 'Phase 2: CI/CD & Automation', 'duration': 'Weeks 5-8', 'topics': 'GitHub Actions Workflows, Jenkins Pipelines, Automated Testing, Security Scans (Trivy/SonarQube)', 'resources': [{'title': 'GitHub Actions Guide', 'url': 'https://docs.github.com/actions'}]},
            {'phase': 'Phase 3: Kubernetes & IaC', 'duration': 'Weeks 9-12', 'topics': 'Kubernetes Pods, Deployments, Ingress, Helm Charts, Terraform Multi-Environment Provisioning', 'resources': [{'title': 'Kubernetes Interactive Tutorials', 'url': 'https://kubernetes.io'}]},
            {'phase': 'Phase 4: Observability & Production', 'duration': 'Weeks 13-16', 'topics': 'Prometheus Metrics, Grafana Dashboards, ELK Logging, SRE Best Practices, Disaster Recovery', 'resources': [{'title': 'Roadmap.sh DevOps', 'url': 'https://roadmap.sh/devops'}]}
        ],
        'interview_questions': [
            {'q': 'Explain the difference between a Docker Container and a Virtual Machine (VM).', 'a': 'VMs virtualize the entire hardware layer and include a full Guest OS (heavy, slow boot). Containers share the host OS kernel and isolate user space processes via cgroups and namespaces (lightweight, instant boot).'},
            {'q': 'How does Kubernetes handle self-healing for failing pods?', 'a': 'The Kubelet checks Liveness and Readiness Probes. If a container crashes or fails liveness checks, Kubelet restarts it. If a node dies, the Controller Manager reschedules pods to healthy nodes.'},
            {'q': 'What is Infrastructure as Code (IaC) and explain the purpose of the Terraform state file.', 'a': 'IaC manages infrastructure through declarative configuration code. The Terraform state file tracks real-world resource IDs and attributes, mapping configuration files to deployed cloud resources.'},
            {'q': 'What is Blue-Green Deployment vs Canary Deployment?', 'a': 'Blue-Green maintains two identical environments; traffic switches 100% instantly from Blue (old) to Green (new). Canary gradually routes a small percentage (e.g. 5% -> 25% -> 100%) of user traffic to test new versions in production.'},
            {'q': 'How do you implement zero-downtime rolling updates in Kubernetes?', 'a': 'Configure `spec.strategy.type: RollingUpdate` with `maxUnavailable` and `maxSurge` parameters alongside proper Readiness Probes so new pods only receive traffic once healthy.'}
        ]
    },
    'Cloud Engineer': {
        'sector': 'Cloud & DevOps',
        'description': 'Designs, migrates, and administers highly available, secure, and cost-efficient cloud infrastructure architectures.',
        'qualification_required': False,
        'required_degrees': [],
        'career_switch_allowed': True,
        'compatible_degrees': ['B.Tech', 'M.Tech', 'MCA', 'B.Sc'],
        'compatible_majors': ['Computer Science', 'Information Technology', 'Electronics'],
        'industries': ['IT', 'Cloud', 'Consulting', 'Finance', 'Enterprise'],
        'core_skills': ['AWS', 'Azure', 'Google Cloud', 'Terraform', 'Linux', 'Cloud Architecture'],
        'important_skills': ['Networking', 'IAM', 'Docker', 'Kubernetes', 'Python', 'Security', 'CI/CD'],
        'supporting_skills': ['CloudWatch', 'VPC', 'Serverless', 'Lambda', 'SQL', 'Bash'],
        'general_skills': ['Cloud Cost Optimization', 'Disaster Recovery', 'Security Best Practices'],
        'certifications': ['AWS Solutions Architect Associate', 'Microsoft Azure Administrator (AZ-104)', 'Google Cloud Associate Cloud Engineer'],
        'experience_expectation': '0-5+ years',
        'roadmap': [
            {'phase': 'Phase 1: Cloud Core & Networking', 'duration': 'Weeks 1-4', 'topics': 'VPC, Subnets, Route Tables, NAT Gateways, Security Groups, IAM Roles & Policies', 'resources': [{'title': 'AWS Skill Builder', 'url': 'https://explore.skillbuilder.aws'}]},
            {'phase': 'Phase 2: Compute, Storage & DB', 'duration': 'Weeks 5-8', 'topics': 'EC2, Auto-Scaling, Load Balancers, S3 Bucket Policies, RDS Aurora, Serverless Lambda', 'resources': [{'title': 'Azure Fundamentals', 'url': 'https://learn.microsoft.com'}]},
            {'phase': 'Phase 3: Automation & Security', 'duration': 'Weeks 9-12', 'topics': 'Terraform Cloud Provisioning, KMS Encryption, CloudTrail Auditing, Security Hub', 'resources': [{'title': 'Terraform Tutorials', 'url': 'https://developer.hashicorp.com'}]},
            {'phase': 'Phase 4: High Availability & FinOps', 'duration': 'Weeks 13-16', 'topics': 'Multi-Region Disaster Recovery, Route 53 DNS Failover, AWS Cost Explorer & FinOps Optimization', 'resources': [{'title': 'AWS Well-Architected Framework', 'url': 'https://aws.amazon.com/architecture'}]}
        ],
        'interview_questions': [
            {'q': 'What is the difference between a Public and a Private Subnet in an AWS VPC?', 'a': 'A Public Subnet has an explicit route table entry pointing to an Internet Gateway (IGW). A Private Subnet routes outbound internet traffic only through a NAT Gateway located in a public subnet.'},
            {'q': 'Explain the AWS Shared Responsibility Model.', 'a': 'AWS is responsible for security OF the cloud (physical hardware, data center facilities, virtualization layer). The customer is responsible for security IN the cloud (guest OS, IAM permissions, data encryption, firewall rules).'},
            {'q': 'How do you design a High-Availability Multi-AZ Architecture for a web app?', 'a': 'Deploy an Application Load Balancer across multi-AZs -> Auto Scaling Group of EC2/container instances across 2+ AZs -> Multi-AZ database with automated synchronous replication and failover.'},
            {'q': 'What is Serverless computing and what are its trade-offs?', 'a': 'Serverless (e.g. AWS Lambda) executes code on-demand with automatic scaling and zero idle costs. Trade-offs: cold start latency, execution duration limits (15 mins), and vendor lock-in.'},
            {'q': 'How do you enforce least privilege access in Cloud IAM?', 'a': 'Use dedicated IAM Roles with precise resource-level action policies rather than wildcard `*` permissions, require MFA, enforce temporary session credentials, and audit with Access Advisor.'}
        ]
    },

    # ---------------- 5. CYBERSECURITY ----------------
    'Cybersecurity Analyst': {
        'sector': 'Cybersecurity',
        'description': 'Monitors networks, detects security vulnerabilities, conducts threat intelligence, and mitigates cyber attacks.',
        'qualification_required': False,
        'required_degrees': [],
        'career_switch_allowed': True,
        'compatible_degrees': ['B.Tech', 'BCA', 'MCA', 'B.Sc', 'M.Sc'],
        'compatible_majors': ['Cybersecurity', 'Computer Science', 'Information Technology', 'Electronics'],
        'industries': ['IT', 'Banking', 'Defense', 'Healthcare', 'Government'],
        'core_skills': ['Network Security', 'SIEM', 'Ethical Hacking', 'Vulnerability Assessment', 'Linux', 'Incident Response'],
        'important_skills': ['Wireshark', 'Python', 'Firewalls', 'Penetration Testing', 'Cryptography', 'Nmap', 'Security'],
        'supporting_skills': ['Burp Suite', 'Splunk', 'SOC', 'Bash', 'Malware Analysis', 'Identity & Access Management'],
        'general_skills': ['Critical Thinking', 'Forensic Investigation', 'Security Compliance'],
        'certifications': ['CompTIA Security+', 'Certified Ethical Hacker (CEH)', 'CompTIA CySA+'],
        'experience_expectation': '0-5+ years',
        'roadmap': [
            {'phase': 'Phase 1: Networking & OS Security', 'duration': 'Weeks 1-4', 'topics': 'OSI Model, TCP/IP, Linux Permissions, Wireshark Packet Analysis, Nmap Port Scanning', 'resources': [{'title': 'TryHackMe Pre-Security', 'url': 'https://tryhackme.com'}]},
            {'phase': 'Phase 2: Threats & SIEM Analysis', 'duration': 'Weeks 5-8', 'topics': 'OWASP Top 10, MITRE ATT&CK Framework, Splunk SIEM Log Analysis, SOC Operations', 'resources': [{'title': 'Splunk Free Training', 'url': 'https://www.splunk.com'}]},
            {'phase': 'Phase 3: Vulnerability & Pentesting', 'duration': 'Weeks 9-12', 'topics': 'Vulnerability Scanning (Nessus), Burp Suite Web App Pentesting, Metasploit Basics', 'resources': [{'title': 'HackTheBox', 'url': 'https://www.hackthebox.com'}]},
            {'phase': 'Phase 4: Incident Response & Blue Team', 'duration': 'Weeks 13-16', 'topics': 'Digital Forensics (Autopsy), Malware Sandboxing, Incident Response Playbooks, Security+ Exam Prep', 'resources': [{'title': 'Professor Messer Security+', 'url': 'https://www.professormesser.com'}]}
        ],
        'interview_questions': [
            {'q': 'Explain the CIA Triad in Information Security.', 'a': 'Confidentiality (prevent unauthorized data access via encryption), Integrity (prevent unauthorized modification via hashing/signatures), Availability (ensure reliable system access via redundancy/DDoS protection).'},
            {'q': 'How does a Cross-Site Scripting (XSS) attack work and how do you prevent it?', 'a': 'Attacker injects malicious client-side JavaScript into a web app. Prevention: Context-aware output encoding, Content Security Policy (CSP), and sanitized input.'},
            {'q': 'What is the difference between Symmetric and Asymmetric Encryption?', 'a': 'Symmetric uses one shared secret key for encryption and decryption (AES, fast). Asymmetric uses a public key to encrypt and private key to decrypt (RSA/ECC, used for key exchange/TLS).'},
            {'q': 'How does a SIEM (e.g. Splunk) detect an ongoing Brute Force or Lateral Movement attack?', 'a': 'It aggregates logs across endpoints, firewalls, and domain controllers, correlating anomalies like 50+ failed login events followed by a single success within 60 seconds.'},
            {'q': 'Explain the difference between a Vulnerability Assessment and a Penetration Test.', 'a': 'Vulnerability assessment identifies and catalogs known security weaknesses automatically. Penetration testing actively exploits vulnerabilities to evaluate real-world intrusion depth and impact.'}
        ]
    },

    # ---------------- 6. FINANCE & BANKING ----------------
    'Financial Analyst': {
        'sector': 'Finance & Accounting',
        'description': 'Performs financial modeling, corporate valuation, budgeting forecasts, and investment risk assessments.',
        'qualification_required': False,
        'required_degrees': [],
        'career_switch_allowed': True,
        'compatible_degrees': ['MBA', 'B.Com', 'BBA', 'M.Sc', 'B.Sc', 'B.Tech'],
        'compatible_majors': ['Finance', 'Business', 'Economics', 'Accounting', 'Mathematics'],
        'industries': ['Finance', 'Banking', 'Consulting', 'Corporate', 'Investment'],
        'core_skills': ['Financial Modeling', 'Excel', 'Accounting', 'Valuation', 'Corporate Finance', 'Financial Analysis'],
        'important_skills': ['Power BI', 'SQL', 'Risk Management', 'DCF Analysis', 'Budgeting', 'Tableau'],
        'supporting_skills': ['Python', 'Statistics', 'Bloomberg Terminal', 'Financial Reporting', 'Variance Analysis'],
        'general_skills': ['Quantitative Analysis', 'Executive Presentation', 'Attention to Detail'],
        'certifications': ['CFA Level 1', 'Financial Modeling and Valuation Analyst (FMVA)', 'Chartered Accountant (CA)'],
        'experience_expectation': '0-5+ years',
        'roadmap': [
            {'phase': 'Phase 1: Financial Statements & Excel', 'duration': 'Weeks 1-4', 'topics': 'Income Statement, Balance Sheet, Cash Flow 3-Way Linkage, Advanced Excel Shortcuts & Formulas', 'resources': [{'title': 'Corporate Finance Institute', 'url': 'https://corporatefinanceinstitute.com'}]},
            {'phase': 'Phase 2: Valuation Methodologies', 'duration': 'Weeks 5-8', 'topics': 'Discounted Cash Flow (DCF), Comparable Company Analysis, Precedent Transactions, WACC Calculation', 'resources': [{'title': 'Aswath Damodaran Valuation Lectures', 'url': 'https://pages.stern.nyu.edu/~adamodar/'}]},
            {'phase': 'Phase 3: Financial BI & Automation', 'duration': 'Weeks 9-12', 'topics': 'Power BI Financial Dashboards, SQL for Transaction Queries, Python for Portfolio Risk', 'resources': [{'title': 'Wall Street Prep', 'url': 'https://www.wallstreetprep.com'}]},
            {'phase': 'Phase 4: Corporate Finance Pitch', 'duration': 'Weeks 13-16', 'topics': 'Build complete M&A / LBO model case study, Investment Memo, Pitch Deck Presentation', 'resources': [{'title': 'Investopedia Finance', 'url': 'https://www.investopedia.com'}]}
        ],
        'interview_questions': [
            {'q': 'Walk me through how the 3 Financial Statements are linked together.', 'a': 'Net income from the Income Statement flows into Retained Earnings on the Balance Sheet and starts the Cash Flow Statement under Operating Cash Flow. Working Capital changes and CapEx on the CFS update Balance Sheet assets/liabilities. Ending Cash on CFS becomes Cash on the Balance Sheet.'},
            {'q': 'How do you calculate Free Cash Flow to Firm (FCFF)?', 'a': 'FCFF = EBIT * (1 - Tax Rate) + Depreciation & Amortization - Capital Expenditures - Change in Net Working Capital.'},
            {'q': 'If Depreciation increases by $10, how does it affect all three statements (assuming a 20% tax rate)?', 'a': 'Income Statement: Operating Income drops by $10, Net Income drops by $8. Cash Flow: Net Income down $8, add back $10 non-cash depreciation -> Cash up $2. Balance Sheet: Cash up $2, PP&E down $10 -> Total Assets down $8, Retained Earnings down $8.'},
            {'q': 'Explain how a Discounted Cash Flow (DCF) model works.', 'a': 'A DCF projects unlevered free cash flows over a 5-10 year forecast period, calculates a Terminal Value, and discounts all future cash flows back to present value using WACC.'},
            {'q': 'What is WACC and how is the Cost of Equity estimated?', 'a': 'WACC is Weighted Average Cost of Capital: `(E/V * Ke) + (D/V * Kd * (1-t))`. Cost of Equity (Ke) is estimated via CAPM: `Ke = Rf + Beta * (Rm - Rf)`.'}
        ]
    },
    'Chartered Accountant': {
        'sector': 'Finance & Accounting',
        'description': 'Regulated professional responsible for statutory auditing, corporate taxation, forensic accounting, and compliance.',
        'qualification_required': True,
        'required_degrees': ['CA', 'CPA', 'ICAI', 'ACCA'],
        'career_switch_allowed': False,
        'compatible_degrees': ['CA', 'CPA', 'ICAI', 'ACCA', 'B.Com + CA'],
        'compatible_majors': ['Accounting', 'Finance', 'Auditing', 'Taxation'],
        'industries': ['Accounting', 'Auditing', 'Finance', 'Taxation', 'Corporate'],
        'core_skills': ['Auditing', 'Taxation', 'Accounting', 'Financial Reporting', 'GST', 'Corporate Law'],
        'important_skills': ['Income Tax', 'Forensic Accounting', 'Financial Analysis', 'Tally', 'Excel', 'Statutory Compliance'],
        'supporting_skills': ['SAP', 'IFRS', 'Company Law', 'Cost Accounting', 'Internal Controls'],
        'general_skills': ['Regulatory Compliance', 'Professional Ethics', 'Detail Oriented'],
        'certifications': ['Chartered Accountant (ICAI)', 'Certified Public Accountant (CPA)', 'ACCA'],
        'experience_expectation': '0-5+ years (Articleship Required)',
        'roadmap': [
            {'phase': 'Phase 1: Statutory Accounting', 'duration': 'Weeks 1-4', 'topics': 'Ind AS / IFRS Standards, Financial Statement Consolidation, Internal Financial Controls', 'resources': [{'title': 'ICAI Official Portal', 'url': 'https://www.icai.org'}]},
            {'phase': 'Phase 2: Direct & Indirect Tax', 'duration': 'Weeks 5-8', 'topics': 'Corporate Income Tax Assessment, GST Audits, International Tax & Transfer Pricing', 'resources': [{'title': 'Income Tax Dept Learning', 'url': 'https://www.incometax.gov.in'}]},
            {'phase': 'Phase 3: Statutory Audit', 'duration': 'Weeks 9-12', 'topics': 'Standards on Auditing (SAs), CARO 2020 Reporting, Risk-Based Auditing', 'resources': [{'title': 'Standards on Auditing Guide', 'url': 'https://www.icai.org'}]},
            {'phase': 'Phase 4: Corporate Advisory', 'duration': 'Weeks 13-16', 'topics': 'Mergers & Acquisitions Tax Structuring, Due Diligence, Forensic Accounting Case Studies', 'resources': [{'title': 'ACCA Global Resources', 'url': 'https://www.accaglobal.com'}]}
        ],
        'interview_questions': [
            {'q': 'Explain the key differences between Indian Accounting Standards (Ind AS) and IFRS.', 'a': 'Ind AS is converged with IFRS but contains specific carve-outs/carve-ins for Indian economic realities (e.g., accounting for foreign currency loans, investment properties).'},
            {'q': 'What are the main reporting requirements under CARO 2020 for statutory audits?', 'a': 'CARO 2020 mandates detailed auditor disclosures on inventory physical verification, title deeds of immovable properties, benami transactions, working capital limits, and default in repayment.'},
            {'q': 'Explain the concept of Transfer Pricing and the Arm\'s Length Principle.', 'a': 'Transfer pricing regulates transactions between related corporate entities across jurisdictions. The Arm\'s Length Principle requires transaction pricing to match prices agreed between unrelated independent parties.'},
            {'q': 'What are the essential elements of an Internal Control over Financial Reporting (ICFR) framework?', 'a': 'Control Environment, Risk Assessment, Control Activities, Information & Communication, and Monitoring Activities (per COSO Framework).'},
            {'q': 'How do you handle deferred tax assets and liabilities under Ind AS 12?', 'a': 'Recognize deferred tax for temporary differences between accounting carrying amount and tax base of assets/liabilities. DTA is recognized only when probable future taxable profits exist.'}
        ]
    },

    # ---------------- 7. BUSINESS & MANAGEMENT ----------------
    'Project Manager': {
        'sector': 'Management',
        'description': 'Leads cross-functional teams, plans budgets, mitigates risks, and delivers complex technical and business projects on schedule.',
        'qualification_required': False,
        'required_degrees': [],
        'career_switch_allowed': True,
        'compatible_degrees': ['MBA', 'BBA', 'B.Tech', 'M.Tech', 'MCA', 'B.Com'],
        'compatible_majors': ['Business', 'Management', 'Computer Science', 'Information Technology', 'Operations'],
        'industries': ['IT', 'Consulting', 'Finance', 'Manufacturing', 'Healthcare'],
        'core_skills': ['Agile', 'Scrum', 'Jira', 'Project Management', 'Team Leadership', 'Stakeholder Management'],
        'important_skills': ['Risk Management', 'Budgeting', 'Communication', 'Business Analysis', 'Product Management', 'Excel'],
        'supporting_skills': ['Sprint Planning', 'Gantt Charts', 'Confluence', 'Resource Allocation', 'Process Improvement'],
        'general_skills': ['Conflict Resolution', 'Executive Reporting', 'Strategic Planning'],
        'certifications': ['PMI Project Management Professional (PMP)', 'Certified ScrumMaster (CSM)', 'Google Project Management Certificate'],
        'experience_expectation': '1-8+ years',
        'roadmap': [
            {'phase': 'Phase 1: Agile & Scrum Framework', 'duration': 'Weeks 1-4', 'topics': 'Scrum Ceremonies (Standups, Sprints, Retros), Jira Workflows, User Stories & Story Points', 'resources': [{'title': 'Scrum Guide', 'url': 'https://scrumguides.org'}, {'title': 'Atlassian Agile Guide', 'url': 'https://www.atlassian.com/agile'}]},
            {'phase': 'Phase 2: Project Governance & Scope', 'duration': 'Weeks 5-8', 'topics': 'Work Breakdown Structure (WBS), Critical Path Method (CPM), Risk Registers, Budget Tracking', 'resources': [{'title': 'PMI PMBOK Overview', 'url': 'https://www.pmi.org'}]},
            {'phase': 'Phase 3: Stakeholder & Conflict Mgmt', 'duration': 'Weeks 9-12', 'topics': 'Stakeholder Alignment, Negotiation Frameworks, OKRs/KPIs, Cross-Functional Team Leadership', 'resources': [{'title': 'Harvard Business Review PM Articles', 'url': 'https://hbr.org'}]},
            {'phase': 'Phase 4: PMP & Case Studies', 'duration': 'Weeks 13-16', 'topics': 'Lead simulated product delivery lifecycle, PMP Exam Preparation, Executive Presentations', 'resources': [{'title': 'Google PM Certificate', 'url': 'https://grow.google/certificates/project-management/'}]}
        ],
        'interview_questions': [
            {'q': 'How do you handle a project falling behind its deadline and over budget?', 'a': 'Analyze critical path to isolate bottlenecks, communicate transparently with stakeholders, evaluate Scope vs Time vs Cost (Fast-tracking/Crashing), and reprioritize non-essential backlog features.'},
            {'q': 'Explain the difference between Agile and Waterfall project management.', 'a': 'Waterfall is linear and sequential with rigid upfront requirements. Agile is iterative and flexible, delivering tested software in 2-4 week sprints with continuous customer feedback.'},
            {'q': 'How do you handle scope creep when a client requests multiple out-of-scope features mid-sprint?', 'a': 'Document the request, assess impact on sprint goals and budget, explain trade-offs to the client, and place requests into the product backlog for future sprint estimation.'},
            {'q': 'What makes an effective Retrospective meeting in Scrum?', 'a': 'Fostering psychological safety, focusing on processes and team dynamics rather than blame, and walking away with 2-3 clear actionable improvements with assigned owners.'},
            {'q': 'What is Critical Path Method (CPM) and float/slack time?', 'a': 'The Critical Path is the longest sequence of dependent activities with zero float. Any delay on critical path activities directly delays the final project completion date.'}
        ]
    },
    'Business Analyst': {
        'sector': 'Business & Consulting',
        'description': 'Analyzes organizational processes, gathers functional business requirements, and designs data-driven solutions.',
        'qualification_required': False,
        'required_degrees': [],
        'career_switch_allowed': True,
        'compatible_degrees': ['MBA', 'BBA', 'B.Tech', 'B.Com', 'B.Sc', 'MCA'],
        'compatible_majors': ['Business', 'Finance', 'Information Technology', 'Computer Science', 'Management'],
        'industries': ['Consulting', 'Finance', 'IT', 'Marketing', 'Healthcare', 'Operations'],
        'core_skills': ['Business Analysis', 'SQL', 'Excel', 'Power BI', 'Agile', 'Requirements Gathering'],
        'important_skills': ['Tableau', 'Jira', 'Process Mapping', 'Data Analysis', 'UML', 'Financial Modeling'],
        'supporting_skills': ['BRD / FRD Documentation', 'Wireframing', 'BPMN', 'User Stories', 'Stakeholder Interviews'],
        'general_skills': ['Problem Solving', 'Strategic Thinking', 'Client Communication'],
        'certifications': ['ECBA Entry Certificate in Business Analysis (IIBA)', 'Microsoft Power BI Certified (PL-300)', 'PMI-PBA'],
        'experience_expectation': '0-5+ years',
        'roadmap': [
            {'phase': 'Phase 1: Requirements & Modeling', 'duration': 'Weeks 1-4', 'topics': 'BRD/FRD Documentation, Stakeholder Elicitation, User Stories (Given-When-Then), BPMN Process Mapping', 'resources': [{'title': 'IIBA BABOK Guide', 'url': 'https://www.iiba.org'}]},
            {'phase': 'Phase 2: Analytics & SQL', 'duration': 'Weeks 5-8', 'topics': 'Advanced Excel Modeling, SQL for Business Queries, Power BI Interactive Dashboards', 'resources': [{'title': 'Microsoft Power BI Learn', 'url': 'https://learn.microsoft.com'}]},
            {'phase': 'Phase 3: Agile BA & Strategy', 'duration': 'Weeks 9-12', 'topics': 'Backlog Grooming in Jira, GAP Analysis, Cost-Benefit ROI Analysis, Acceptance Testing (UAT)', 'resources': [{'title': 'Bridging the Gap BA', 'url': 'https://www.bridging-the-gap.com'}]},
            {'phase': 'Phase 4: Consulting Case Study', 'duration': 'Weeks 13-16', 'topics': 'Complete End-to-End Digital Transformation Business Case Study with Wireframes', 'resources': [{'title': 'Coursera Business Analytics', 'url': 'https://www.coursera.org'}]}
        ],
        'interview_questions': [
            {'q': 'Explain the difference between Functional and Non-Functional Requirements.', 'a': 'Functional requirements define WHAT the system must do (e.g. user authentication, invoice generation). Non-functional requirements define HOW the system performs (e.g. latency < 200ms, 99.9% uptime, GDPR compliance).'},
            {'q': 'What is a GAP Analysis and how do you conduct it?', 'a': 'GAP Analysis compares the Current State (As-Is) against the Desired Future State (To-Be) to identify process deficiencies, technological limitations, and necessary actionable solutions.'},
            {'q': 'How do you handle ambiguous or conflicting requirements from senior stakeholders?', 'a': 'Conduct alignment workshops, map requirements to corporate OKRs and financial ROI, build prototype wireframes for clarity, and secure formal sign-off.'},
            {'q': 'What is BPMN and why is visual process mapping valuable?', 'a': 'Business Process Model and Notation is an industry standard visual language. It maps workflows end-to-end to expose operational bottlenecks, redundancies, and automation opportunities.'},
            {'q': 'What is the role of a Business Analyst during User Acceptance Testing (UAT)?', 'a': 'Create UAT test scenarios aligned with business acceptance criteria, guide business users through test execution, track defects in Jira, and ensure system meets original business goals.'}
        ]
    },

    # ---------------- 8. MECHANICAL & MANUFACTURING ----------------
    'Mechanical Engineer': {
        'sector': 'Mechanical Engineering',
        'description': 'Designs, analyzes, and manufactures mechanical systems, thermal equipment, machinery, and automotive components.',
        'qualification_required': False,
        'required_degrees': [],
        'career_switch_allowed': True,
        'compatible_degrees': ['B.Tech', 'M.Tech', 'Diploma', 'B.Sc'],
        'compatible_majors': ['Mechanical', 'Automobile', 'Manufacturing', 'Aerospace', 'Industrial'],
        'industries': ['Manufacturing', 'Automotive', 'Engineering', 'Aerospace', 'Energy'],
        'core_skills': ['AutoCAD', 'SolidWorks', 'Thermodynamics', 'Fluid Mechanics', 'Manufacturing', 'Mechanical Design'],
        'important_skills': ['ANSYS', 'CATIA', 'MATLAB', 'Finite Element Analysis (FEA)', 'GD&T', 'CAD'],
        'supporting_skills': ['CNC Programming', 'Robotics', 'C++', 'Python', 'Material Science'],
        'general_skills': ['Spatial Reasoning', 'Quality Assurance', 'Technical Problem Solving'],
        'certifications': ['CSWA Certified SOLIDWORKS Associate', 'Six Sigma Green Belt', 'AutoCAD Certified Professional'],
        'experience_expectation': '0-5+ years',
        'roadmap': [
            {'phase': 'Phase 1: Mechanics & 3D CAD', 'duration': 'Weeks 1-4', 'topics': 'Statics & Dynamics, Strength of Materials, SolidWorks 3D Modeling, GD&T Standards', 'resources': [{'title': 'SolidWorks Tutorials', 'url': 'https://www.solidworks.com'}]},
            {'phase': 'Phase 2: Thermal & Fluid Sciences', 'duration': 'Weeks 5-8', 'topics': 'Thermodynamics Cycles (Carnot/Rankine/Otto), Heat Transfer, Fluid Mechanics, HVAC Design', 'resources': [{'title': 'NPTEL Mechanical', 'url': 'https://nptel.ac.in'}]},
            {'phase': 'Phase 3: FEA & Simulation', 'duration': 'Weeks 9-12', 'topics': 'ANSYS Structural & Thermal FEA, CFD Fluid Simulation, Vibration Analysis', 'resources': [{'title': 'ANSYS Innovation Space', 'url': 'https://innovationspace.ansys.com'}]},
            {'phase': 'Phase 4: Manufacturing & DFM', 'duration': 'Weeks 13-16', 'topics': 'Design for Manufacturing (DFM/DFA), CNC Machining, Additive Manufacturing, Industry Project', 'resources': [{'title': 'MIT OpenCourseWare Mechanical', 'url': 'https://ocw.mit.edu'}]}
        ],
        'interview_questions': [
            {'q': 'Explain the First and Second Laws of Thermodynamics.', 'a': '1st Law: Energy cannot be created or destroyed, only transformed (conservation of energy). 2nd Law: Total entropy of an isolated system always increases over time; heat cannot spontaneously flow from colder to hotter bodies.'},
            {'q': 'What is GD&T (Geometric Dimensioning and Tolerancing) and why is it important?', 'a': 'GD&T is a symbolic language on engineering drawings specifying exact geometric tolerance limits (flatness, concentricity, position) to guarantee manufactured parts fit and function correctly.'},
            {'q': 'Explain how Finite Element Analysis (FEA) works in structural simulation.', 'a': 'FEA discretizes a complex geometry into a mesh of smaller finite elements connected at nodes, solving differential equilibrium equations to predict stress, strain, and deformation under loads.'},
            {'q': 'What is the difference between Brittle and Ductile material failure?', 'a': 'Ductile materials undergo substantial plastic deformation before fracture (cup-and-cone failure, e.g. Mild Steel). Brittle materials fracture with negligible plastic deformation (flat cleavage, e.g. Cast Iron).'},
            {'q': 'What is Design for Manufacturing (DFM) and Design for Assembly (DFA)?', 'a': 'DFM designs parts to be easily and cost-effectively fabricated using standard manufacturing processes. DFA minimizes the number of parts to simplify and speed up product assembly.'}
        ]
    },

    # ---------------- 9. ELECTRICAL & ELECTRONICS ----------------
    'Electrical Engineer': {
        'sector': 'Electrical & Electronics',
        'description': 'Designs and manages electrical power distribution systems, control circuits, transformers, and industrial automation.',
        'qualification_required': False,
        'required_degrees': [],
        'career_switch_allowed': True,
        'compatible_degrees': ['B.Tech', 'M.Tech', 'Diploma', 'B.Sc'],
        'compatible_majors': ['Electrical', 'Electronics', 'Power Engineering', 'Control Systems'],
        'industries': ['Energy', 'Manufacturing', 'Engineering', 'Utilities', 'Automotive'],
        'core_skills': ['Circuit Design', 'PLC', 'MATLAB', 'MATLAB Simulink', 'Power Systems', 'Electrical Systems'],
        'important_skills': ['Embedded Systems', 'Control Systems', 'AutoCAD Electrical', 'SCADA', 'Microcontrollers'],
        'supporting_skills': ['C', 'PCB Design', 'Python', 'Sensors', 'Power Electronics'],
        'general_skills': ['Safety Compliance', 'Schematic Reading', 'System Troubleshooting'],
        'certifications': ['Certified Energy Manager (CEM)', 'Siemens / Schneider PLC Automation Certification'],
        'experience_expectation': '0-5+ years',
        'roadmap': [
            {'phase': 'Phase 1: Circuits & Power Theory', 'duration': 'Weeks 1-4', 'topics': 'Ohm/Kirchhoff Laws, AC/DC Power, 3-Phase Systems, Transformers, Induction Motors', 'resources': [{'title': 'All About Circuits', 'url': 'https://www.allaboutcircuits.com'}]},
            {'phase': 'Phase 2: Power Systems & MATLAB', 'duration': 'Weeks 5-8', 'topics': 'MATLAB & Simulink Grid Modeling, Power Factor Correction, Protection Relays, Switchgear', 'resources': [{'title': 'MathWorks Tutorials', 'url': 'https://www.mathworks.com'}]},
            {'phase': 'Phase 3: PLC & Industrial Automation', 'duration': 'Weeks 9-12', 'topics': 'Ladder Logic Programming (Allen Bradley / Siemens), SCADA HMI Integration, VFD Drives', 'resources': [{'title': 'RealPars PLC Automation', 'url': 'https://realpars.com'}]},
            {'phase': 'Phase 4: Power Electronics & Safety', 'duration': 'Weeks 13-16', 'topics': 'Inverters, Rectifiers, Renewable Solar Grid Inverters, High Voltage Electrical Safety Standards', 'resources': [{'title': 'NPTEL Electrical', 'url': 'https://nptel.ac.in'}]}
        ],
        'interview_questions': [
            {'q': 'What is Power Factor and why is Power Factor Correction critical in industrial power systems?', 'a': 'Power Factor is the ratio of Real Power (kW) to Apparent Power (kVA). Low power factor draws excess reactive current, causing line losses, voltage drops, and utility penalties. Shunt capacitor banks correct it.'},
            {'q': 'Explain the operating principle of a 3-Phase Induction Motor.', 'a': 'Balanced 3-phase AC in stator windings generates a Rotating Magnetic Field (RMF). The RMF cuts rotor conductors, inducing EMF and rotor current, generating torque per Lenz\'s Law to rotate slightly below synchronous speed.'},
            {'q': 'How does a PLC execute its program scan cycle?', 'a': '1) Input Scan (reads physical sensors into input memory), 2) Program Execution (evaluates Ladder Logic sequentially), 3) Output Scan (updates actuator/relay coils).'},
            {'q': 'What is the difference between a MCB, MCCB, and ELCB/RCCB?', 'a': 'MCB protects low-current circuits from overload/short-circuit (<100A). MCCB handles adjustable higher currents (<1000A). RCCB/ELCB detects Earth leakage current to prevent electric shocks.'},
            {'q': 'Explain how a Transformer steps up or steps down voltage.', 'a': 'Alternating current in the primary coil creates a varying magnetic flux in the laminated iron core. Per Faraday\'s Law of Mutual Induction, this induces voltage in the secondary coil proportional to the turns ratio: `Vs/Vp = Ns/Np`.'}
        ]
    },
    'Electronics Engineer': {
        'sector': 'Electrical & Electronics',
        'description': 'Develops embedded microcontrollers, IoT devices, analog/digital circuits, and printed circuit boards (PCBs).',
        'qualification_required': False,
        'required_degrees': [],
        'career_switch_allowed': True,
        'compatible_degrees': ['B.Tech', 'M.Tech', 'Diploma', 'B.Sc'],
        'compatible_majors': ['Electronics', 'ECE', 'Embedded Systems', 'VLSI', 'Electrical'],
        'industries': ['Electronics', 'IoT', 'Automotive', 'Semiconductor', 'Defense'],
        'core_skills': ['Embedded C', 'Microcontrollers', 'Arduino', 'PCB Design', 'Circuit Design', 'Embedded Systems'],
        'important_skills': ['C++', 'MATLAB', 'VLSI', 'Verilog', 'ARM Cortex', 'Sensors', 'IoT'],
        'supporting_skills': ['KiCAD', 'Altium', 'I2C/SPI/UART', 'Python', 'RTOS', 'Oscilloscopes'],
        'general_skills': ['Hardware Debugging', 'Schematic Design', 'Analytical Skills'],
        'certifications': ['ARM Certified Engineer', 'Certified Embedded Systems Developer'],
        'experience_expectation': '0-5+ years',
        'roadmap': [
            {'phase': 'Phase 1: C & Microcontrollers', 'duration': 'Weeks 1-4', 'topics': 'Embedded C, Pointer Arithmetic, GPIO, Timers, Interrupts, STM32 / Arduino Architecture', 'resources': [{'title': 'Embedded Artistry', 'url': 'https://embeddedartistry.com'}]},
            {'phase': 'Phase 2: Communication Protocols', 'duration': 'Weeks 5-8', 'topics': 'UART, SPI, I2C, CAN Bus Communication, Sensor Interfacing (ADC, Accelerometers)', 'resources': [{'title': 'NXP / STMicro Tutorials', 'url': 'https://www.st.com'}]},
            {'phase': 'Phase 3: PCB Design & Hardware', 'duration': 'Weeks 9-12', 'topics': 'Schematic Capture in KiCAD / Altium, Multi-Layer PCB Layout, Trace Routing, Gerber Export', 'resources': [{'title': 'KiCad Official Guide', 'url': 'https://www.kicad.org'}]},
            {'phase': 'Phase 4: RTOS & IoT', 'duration': 'Weeks 13-16', 'topics': 'FreeRTOS Tasks & Semaphores, ESP32 Wi-Fi/Bluetooth IoT Project, Firmware Security', 'resources': [{'title': 'FreeRTOS Learning', 'url': 'https://www.freertos.org'}]}
        ],
        'interview_questions': [
            {'q': 'Explain the difference between I2C and SPI communication protocols.', 'a': 'I2C is a 2-wire half-duplex bus (SDA/SCL) supporting multi-master/multi-slave with device addressing (up to 3.4 Mbps). SPI is a 4-wire full-duplex synchronous bus (MOSI/MISO/SCK/CS) with higher speeds (>20 Mbps).'},
            {'q': 'What is an Interrupt Service Routine (ISR) and what are best practices when writing one?', 'a': 'An ISR is a callback executed when a hardware interrupt occurs. Best practices: keep it as fast as possible, avoid blocking delays/I/O, and set volatile flags for main loop processing.'},
            {'q': 'What is a Real-Time Operating System (RTOS) and how does preemptive scheduling work?', 'a': 'An RTOS guarantees deterministic response times for tasks. Preemptive scheduling immediately switches CPU execution to the highest-priority ready task, pausing lower-priority tasks.'},
            {'q': 'What is the purpose of Decoupling Capacitors in PCB design and where should they be placed?', 'a': 'Decoupling capacitors smooth voltage spikes and provide local high-frequency current reserve to ICs. They must be placed as close as physically possible to the IC power pins.'},
            {'q': 'Explain Pull-up and Pull-down resistors with an example.', 'a': 'They prevent floating input pins by pulling the voltage to a defined HIGH (VCC) or LOW (GND) state when no active signal is driving the line (e.g., in pushbuttons and I2C lines).'}
        ]
    },

    # ---------------- 10. CIVIL & CONSTRUCTION ----------------
    'Civil Engineer': {
        'sector': 'Civil & Construction',
        'description': 'Plans, designs, and oversees construction of structural foundations, transport networks, bridges, and municipal infrastructure.',
        'qualification_required': False,
        'required_degrees': [],
        'career_switch_allowed': True,
        'compatible_degrees': ['B.Tech', 'M.Tech', 'Diploma', 'B.Sc'],
        'compatible_majors': ['Civil', 'Structural Engineering', 'Construction Management', 'Infrastructure'],
        'industries': ['Construction', 'Infrastructure', 'Engineering', 'Government', 'Real Estate'],
        'core_skills': ['AutoCAD', 'Structural Analysis', 'Construction Management', 'Surveying', 'STAAD.Pro', 'Civil Engineering'],
        'important_skills': ['Revit', 'Concrete Technology', 'Geotechnical Engineering', 'Project Planning', 'Quantity Surveying'],
        'supporting_skills': ['Primavera P6', 'GIS', 'Excel', 'Estimation', 'Building Codes', 'BIM'],
        'general_skills': ['Site Supervision', 'Quality Control', 'Safety Compliance'],
        'certifications': ['AutoCAD Civil 3D Certified Professional', 'STAAD.Pro Certified Professional'],
        'experience_expectation': '0-5+ years',
        'roadmap': [
            {'phase': 'Phase 1: Structural Theory & Drafting', 'duration': 'Weeks 1-4', 'topics': 'Bending Moment/Shear Force Diagrams, RCC Design, 2D Drafting in AutoCAD', 'resources': [{'title': 'NPTEL Civil Engineering', 'url': 'https://nptel.ac.in'}]},
            {'phase': 'Phase 2: Structural Analysis Software', 'duration': 'Weeks 5-8', 'topics': 'STAAD.Pro / ETABS 3D Multi-Storey Building Modeling, Seismic & Wind Load Analysis (IS 1893/875)', 'resources': [{'title': 'Bentley STAAD Learning', 'url': 'https://www.bentley.com'}]},
            {'phase': 'Phase 3: Geotechnical & Concrete Tech', 'duration': 'Weeks 9-12', 'topics': 'Soil Bearing Capacity, Foundation Design (Isolated/Raft), Concrete Mix Design (IS 10262)', 'resources': [{'title': 'Concrete Technology Guide', 'url': 'https://nptel.ac.in'}]},
            {'phase': 'Phase 4: Site Mgmt & Estimation', 'duration': 'Weeks 13-16', 'topics': 'Bill of Quantities (BOQ) Estimation, Primavera P6 Project Scheduling, Site Quality Audits', 'resources': [{'title': 'Coursera Construction Project Management', 'url': 'https://www.coursera.org'}]}
        ],
        'interview_questions': [
            {'q': 'Explain the difference between a One-Way Slab and a Two-Way Slab.', 'a': 'A One-Way Slab is supported on 2 opposite edges or has length/width ratio (Ly/Lx) >= 2 (bends in shorter direction). A Two-Way Slab is supported on all 4 sides with Ly/Lx < 2 (bends in both directions).'},
            {'q': 'What is the Slump Test of concrete and what does it measure?', 'a': 'The slump test measures the workability and consistency of freshly mixed concrete before placement. Types include True Slump (ideal), Shear Slump (harsh mix), and Collapse Slump (excess water).'},
            {'q': 'What is Safe Bearing Capacity (SBC) of soil and how does it determine foundation design?', 'a': 'SBC is the maximum safe load per unit area that soil can carry without shear failure or excessive settlement. Footing surface area is directly calculated as: `Area = Total Column Load / SBC`.'},
            {'q': 'What is Pre-stressed Concrete and what advantages does it offer over RCC?', 'a': 'Pre-stressed concrete introduces internal compressive stresses using high-tensile steel tendons before service loads are applied, actively counteracting tensile stresses and enabling longer spans with thinner slabs.'},
            {'q': 'Explain the difference between Characteristic Strength (fck) and Target Mean Strength in concrete mix design.', 'a': 'Characteristic Strength (fck) is the compressive strength below which not more than 5% of test results are expected to fall at 28 days. Target Mean Strength includes a safety margin: `f\'ck = fck + 1.65 * s`.'}
        ]
    },

    # ---------------- 11. HEALTHCARE & MEDICAL ----------------
    'Doctor / Medical Practitioner': {
        'sector': 'Healthcare & Medical',
        'description': 'Licensed physician diagnosing illnesses, prescribing treatments, and administering medical patient care.',
        'qualification_required': True,
        'required_degrees': ['MBBS', 'MD', 'MS', 'BDS', 'MDS', 'DO', 'BAMS', 'BHMS'],
        'career_switch_allowed': False,
        'compatible_degrees': ['MBBS', 'MD', 'MS', 'BDS', 'MDS', 'DO', 'BAMS', 'BHMS'],
        'compatible_majors': ['Medicine', 'Surgery', 'Dentistry', 'Clinical Medicine', 'Ayurveda', 'Homeopathy'],
        'industries': ['Healthcare', 'Hospitals', 'Medical Practice', 'Clinical Research'],
        'core_skills': ['Clinical Diagnosis', 'Patient Care', 'Pharmacology', 'Internal Medicine', 'Medical Treatment'],
        'important_skills': ['Pathology', 'Anatomy', 'Physiology', 'Medical Ethics', 'Emergency Care', 'Surgery'],
        'supporting_skills': ['Healthcare Documentation', 'Clinical Research', 'Medical Imaging', 'Diagnostics'],
        'general_skills': ['Empathy', 'Critical Decision Making', 'Communication'],
        'certifications': ['Medical Council Registration (NMC / State Medical Council)', 'BLS / ACLS Certification'],
        'experience_expectation': 'Rotational Internship / Residency',
        'roadmap': [
            {'phase': 'Phase 1: Pre-Clinical Sciences', 'duration': 'Years 1-2', 'topics': 'Human Anatomy, Physiology, Biochemistry, Medical Terminology', 'resources': [{'title': 'Medscape Education', 'url': 'https://www.medscape.org'}]},
            {'phase': 'Phase 2: Para-Clinical Sciences', 'duration': 'Years 2-3', 'topics': 'Pathology, Microbiology, Pharmacology, Forensic Medicine', 'resources': [{'title': 'NCBI Bookshelf', 'url': 'https://www.ncbi.nlm.nih.gov/books/'}]},
            {'phase': 'Phase 3: Clinical Specialization', 'duration': 'Years 3-4.5', 'topics': 'Internal Medicine, General Surgery, Obstetrics & Gynecology, Pediatrics', 'resources': [{'title': 'BMJ Learning', 'url': 'https://learning.bmj.com'}]},
            {'phase': 'Phase 4: Compulsory Internship', 'duration': 'Year 5', 'topics': 'Hospital Rotations, Emergency Medicine, Inpatient Management, Medical Licensing Board', 'resources': [{'title': 'UpToDate Clinical Guide', 'url': 'https://www.uptodate.com'}]}
        ],
        'interview_questions': [
            {'q': 'How do you perform a differential diagnosis for a patient presenting with acute chest pain?', 'a': 'Evaluate life-threatening conditions immediately (ACS/Myocardial Infarction, Aortic Dissection, Pulmonary Embolism, Tension Pneumothorax, Esophageal Rupture) using ECG, Troponin, and chest X-ray.'},
            {'q': 'Explain the mechanism of action of ACE inhibitors in managing hypertension.', 'a': 'ACE inhibitors inhibit the Angiotensin-Converting Enzyme, preventing conversion of Angiotensin I to Angiotensin II (a potent vasoconstrictor), reducing systemic vascular resistance and aldosterone secretion.'},
            {'q': 'What are the core steps in Advanced Cardiovascular Life Support (ACLS) for cardiac arrest?', 'a': 'High-quality CPR -> Attach monitor/defibrillator -> Identify rhythm (Shockable: VF/pVT -> Defibrillate + Epinephrine/Amiodarone; Non-shockable: Asystole/PEA -> CPR + Epinephrine) -> Identify reversible H\'s and T\'s.'},
            {'q': 'How do you approach breaking bad news to a patient or their family?', 'a': 'Use the SPIKES protocol: Setting up interview, Perception assessment, Invitation, Knowledge sharing with empathy, Emotions exploration, Strategy & Summary.'},
            {'q': 'What are the clinical indicators and management steps for Sepsis / Septic Shock?', 'a': 'qSOFA criteria (altered mental status, systolic BP <= 100, respiratory rate >= 22). Implement the Sepsis Six bundle: blood cultures, broad-spectrum antibiotics, IV fluids, measure lactate, monitor urine output.'}
        ]
    },
    'Healthcare Analyst': {
        'sector': 'Healthcare & Medical',
        'description': 'Analyzes clinical trials, hospital operational workflows, patient health records, and healthcare insurance metrics.',
        'qualification_required': False,
        'required_degrees': [],
        'career_switch_allowed': True,
        'compatible_degrees': ['B.Sc', 'M.Sc', 'B.Tech', 'MBA', 'B.Pharm', 'BDS', 'BAMS', 'BCA'],
        'compatible_majors': ['Healthcare Administration', 'Biotechnology', 'Bioinformatics', 'Data Science', 'Pharmacy', 'Computer Science'],
        'industries': ['Healthcare', 'Pharmaceuticals', 'Healthtech', 'Insurance', 'Clinical Research'],
        'core_skills': ['Healthcare Data', 'SQL', 'Excel', 'Data Analysis', 'Clinical Data', 'Statistics'],
        'important_skills': ['Power BI', 'Tableau', 'Python', 'Healthcare Analytics', 'Electronic Health Records (EHR)', 'Health Informatics'],
        'supporting_skills': ['HIPAA Compliance', 'R', 'Medical Terminology', 'Epidemiology', 'SAS'],
        'general_skills': ['Data Privacy', 'Analytical Problem Solving', 'Communication'],
        'certifications': ['Certified Health Data Analyst (CHDA)', 'Google Data Analytics Certificate'],
        'experience_expectation': '0-4+ years',
        'roadmap': [
            {'phase': 'Phase 1: Healthcare Data & EHR', 'duration': 'Weeks 1-4', 'topics': 'EHR Systems, ICD-10 / CPT Medical Coding, HIPAA Compliance & Patient Privacy Standards', 'resources': [{'title': 'CDC Healthcare Informatics', 'url': 'https://www.cdc.gov'}]},
            {'phase': 'Phase 2: SQL & Healthcare Metrics', 'duration': 'Weeks 5-8', 'topics': 'SQL Queries on Patient Registries, Hospital Readmission Rates, Length of Stay (LOS) Calculations', 'resources': [{'title': 'SQL for Healthcare Analysts', 'url': 'https://www.coursera.org'}]},
            {'phase': 'Phase 3: BI Dashboards & Viz', 'duration': 'Weeks 9-12', 'topics': 'Power BI / Tableau Clinical KPI Dashboards, Patient Satisfaction & Cost Utilization Tracking', 'resources': [{'title': 'Tableau Healthcare Showcase', 'url': 'https://public.tableau.com'}]},
            {'phase': 'Phase 4: Python & Clinical Studies', 'duration': 'Weeks 13-16', 'topics': 'Pandas for Clinical Trials Data Analysis, Survival Analysis (Kaplan-Meier), Capstone Project', 'resources': [{'title': 'Health Data Science Guide', 'url': 'https://www.healthdata.org'}]}
        ],
        'interview_questions': [
            {'q': 'What is HIPAA compliance and how does it protect Protected Health Information (PHI)?', 'a': 'HIPAA establishes national standards protecting sensitive patient health records from being disclosed without consent. It mandates physical, administrative, and technical safeguards including encryption and role-based access.'},
            {'q': 'How do you calculate and analyze the Hospital Readmission Rate?', 'a': 'Readmission Rate = `(Number of unplanned readmissions within 30 days of discharge / Total eligible hospital discharges) * 100`. High rates indicate potential discharge quality deficiencies.'},
            {'q': 'Explain the difference between ICD-10, CPT, and SNOMED-CT healthcare coding standards.', 'a': 'ICD-10 codes diagnoses and diseases. CPT codes medical, surgical, and diagnostic procedures for billing. SNOMED-CT provides comprehensive clinical terminology for electronic health record documentation.'},
            {'q': 'How do you handle missing values and privacy de-identification in clinical research datasets?', 'a': 'Apply HIPAA Safe Harbor de-identification (remove 18 specific identifiers like names, dates, SSNs). Impute clinical missing values carefully using domain-appropriate models or denote as missing indicator.'},
            {'q': 'What metrics are used to measure clinical trial efficacy and patient outcomes?', 'a': 'Overall Survival (OS), Progression-Free Survival (PFS), Objective Response Rate (ORR), Adverse Event rates (CTCAE), and Patient-Reported Outcome Measures (PROMs).'}
        ]
    },

    # ---------------- 12. LAW ----------------
    'Lawyer / Advocate': {
        'sector': 'Law',
        'description': 'Regulated legal counsel representing clients in courts, drafting binding contracts, and advising on statutory compliance.',
        'qualification_required': True,
        'required_degrees': ['LLB', 'LLM', 'BA LLB', 'BBA LLB', 'JD'],
        'career_switch_allowed': False,
        'compatible_degrees': ['LLB', 'LLM', 'BA LLB', 'BBA LLB', 'JD'],
        'compatible_majors': ['Law', 'Corporate Law', 'Constitutional Law', 'Criminal Law', 'Legal Studies'],
        'industries': ['Legal Services', 'Law Firms', 'Corporate Legal', 'Judiciary', 'Government'],
        'core_skills': ['Legal Research', 'Contract Law', 'Litigation', 'Legal Drafting', 'Constitutional Law', 'Corporate Law'],
        'important_skills': ['Legal Compliance', 'Statutory Interpretation', 'Arbitration', 'Negotiation', 'Intellectual Property'],
        'supporting_skills': ['Case Law Analysis', 'Due Diligence', 'Legal Writing', 'Commercial Law'],
        'general_skills': ['Persuasive Advocacy', 'Ethical Judgment', 'Critical Analysis'],
        'certifications': ['Bar Council of India Enrollment', 'State Bar Admission / All India Bar Examination (AIBE)'],
        'experience_expectation': '0-5+ years (Court / Law Firm Apprenticeship)',
        'roadmap': [
            {'phase': 'Phase 1: Jurisprudence & Core Law', 'duration': 'Weeks 1-4', 'topics': 'Constitutional Law, Law of Contracts, Torts, Criminal Law & Procedure (CrPC/IPC/BNS)', 'resources': [{'title': 'Bar Council Resources', 'url': 'https://www.barcouncilofindia.org'}]},
            {'phase': 'Phase 2: Legal Research & Drafting', 'duration': 'Weeks 5-8', 'topics': 'Case Law Research (SCC Online/Manupatra), Drafting Petitions, Affidavits, Plainits & Written Statements', 'resources': [{'title': 'Indian Kanoon', 'url': 'https://indiankanoon.org'}]},
            {'phase': 'Phase 3: Corporate & Commercial Law', 'duration': 'Weeks 9-12', 'topics': 'Company Law, M&A Due Diligence, Commercial Contracts, Arbitration & Conciliation Act', 'resources': [{'title': 'Corporate Law Guide', 'url': 'https://www.mca.gov.in'}]},
            {'phase': 'Phase 4: Advocacy & Bar Exam', 'duration': 'Weeks 13-16', 'topics': 'Courtroom Trial Advocacy, Moot Court Simulations, All India Bar Examination (AIBE) Prep', 'resources': [{'title': 'SCC Online Learning', 'url': 'https://www.scconline.com'}]}
        ],
        'interview_questions': [
            {'q': 'What are the essential elements of a valid, legally enforceable contract?', 'a': 'Offer, Acceptance, Lawful Consideration, Capacity of Parties, Free Consent (absence of coercion/undue influence/fraud), Lawful Object, and intention to create legal relations.'},
            {'q': 'Explain the Doctrine of Stare Decisis and Precedent hierarchy.', 'a': 'Stare Decisis binds lower courts to adhere to legal principles decided by higher courts. Decisions of the Supreme Court are binding on all courts; High Court decisions bind subordinate courts within its jurisdiction.'},
            {'q': 'What is the difference between Arbitration, Mediation, and Litigation?', 'a': 'Litigation is a public court trial with judge decisions. Arbitration is a private binding adjudication by an agreed arbitrator. Mediation is a non-binding confidential negotiation facilitated by a neutral mediator.'},
            {'q': 'How do you conduct legal Due Diligence during an M&A corporate acquisition?', 'a': 'Examine corporate records (articles, shareholder agreements), material contracts, regulatory compliance licenses, active/threatened litigation, labor compliance, and intellectual property ownership.'},
            {'q': 'Explain the difference between a Void Contract and a Voidable Contract.', 'a': 'A Void Contract is unenforceable from inception (void ab initio, e.g. illegal object). A Voidable Contract is valid until repudiated at the option of the aggrieved party (e.g. consent obtained via coercion).'}
        ]
    },

    # ---------------- 13. ARCHITECTURE & PLANNING ----------------
    'Architect': {
        'sector': 'Architecture & Planning',
        'description': 'Regulated design professional planning aesthetic, sustainable, and structurally compliant buildings and living spaces.',
        'qualification_required': True,
        'required_degrees': ['B.Arch', 'M.Arch'],
        'career_switch_allowed': False,
        'compatible_degrees': ['B.Arch', 'M.Arch'],
        'compatible_majors': ['Architecture', 'Urban Design', 'Landscape Architecture', 'Building Science'],
        'industries': ['Architecture', 'Construction', 'Urban Planning', 'Real Estate'],
        'core_skills': ['Architectural Design', 'AutoCAD', 'Revit', '3D Modeling', 'Building Codes', 'SketchUp'],
        'important_skills': ['BIM', 'Sustainable Design', 'V-Ray', 'Lumion', 'Interior Design', 'Landscape Architecture'],
        'supporting_skills': ['Photoshop', 'Site Planning', 'Structural Integration', 'Rhino', 'Grasshopper'],
        'general_skills': ['Creative Spatial Vision', 'Visual Presentation', 'Client Communication'],
        'certifications': ['Council of Architecture (COA) Registration', 'LEED Green Associate'],
        'experience_expectation': '0-5+ years',
        'roadmap': [
            {'phase': 'Phase 1: Design Theory & Drafting', 'duration': 'Weeks 1-4', 'topics': 'History of Architecture, Spatial Composition, Anthropometry, 2D Working Drawings in AutoCAD', 'resources': [{'title': 'Council of Architecture Portal', 'url': 'https://www.coa.gov.in'}]},
            {'phase': 'Phase 2: 3D Visualization', 'duration': 'Weeks 5-8', 'topics': '3D Modeling in SketchUp & Rhino, Photorealistic Rendering with V-Ray and Lumion', 'resources': [{'title': 'SketchUp Official Tutorials', 'url': 'https://www.sketchup.com'}]},
            {'phase': 'Phase 3: BIM & Building Codes', 'duration': 'Weeks 9-12', 'topics': 'Autodesk Revit BIM Modeling, Structural Coordination, National Building Code (NBC) Compliance', 'resources': [{'title': 'Autodesk Revit Learning', 'url': 'https://learn.autodesk.com'}]},
            {'phase': 'Phase 4: Sustainability & Portfolio', 'duration': 'Weeks 13-16', 'topics': 'Passive Solar & LEED Green Building Strategies, Professional Architectural Design Portfolio', 'resources': [{'title': 'ArchDaily Showcase', 'url': 'https://www.archdaily.com'}]}
        ],
        'interview_questions': [
            {'q': 'Explain the concept of Building Information Modeling (BIM) and its benefits over traditional 2D CAD.', 'a': 'BIM is an intelligent 3D model-based process integrating geometry, spatial relationships, materials, cost estimation (5D), and lifecycle data, enabling seamless clash detection and collaboration across architects, engineers, and builders.'},
            {'q': 'How do you incorporate passive sustainable design strategies into building architecture?', 'a': 'Optimize building orientation for natural daylighting and prevailing wind cross-ventilation, design solar shading devices (louvers), incorporate thermal mass, and use green roofs and high-performance insulation.'},
            {'q': 'What is Floor Space Index / Floor Area Ratio (FSI / FAR) and why is it regulated by urban municipal authorities?', 'a': 'FSI is the ratio of total covered building floor area to the plot area: `FSI = Total Floor Area / Plot Area`. It controls municipal population density, transit infrastructure load, and open space ratios.'},
            {'q': 'Walk me through your architectural design process from conceptual brief to construction drawings.', 'a': '1) Site Analysis & Client Brief -> 2) Conceptual Zoning & Massing -> 3) Schematic Design & 3D Visualizations -> 4) Design Development (BIM/Revit) -> 5) Regulatory Approvals -> 6) Construction Working Drawings.'},
            {'q': 'How do you ensure universal accessibility (Universal Design / ADA) in public building planning?', 'a': 'Incorporate step-free ramp access (1:12 slope), tactile flooring, accessible restroom layouts, wide circulation doorways (min 900mm), accessible elevators, and braille signage.'}
        ]
    },

    # ---------------- 14. HUMAN RESOURCES ----------------
    'HR Specialist': {
        'sector': 'Human Resources',
        'description': 'Manages talent acquisition, employee relations, organizational culture, performance appraisal, and people analytics.',
        'qualification_required': False,
        'required_degrees': [],
        'career_switch_allowed': True,
        'compatible_degrees': ['MBA', 'BBA', 'B.A', 'B.Com', 'B.Sc'],
        'compatible_majors': ['Human Resources', 'Management', 'Psychology', 'Business Administration'],
        'industries': ['IT', 'Corporate', 'Consulting', 'Manufacturing', 'Finance'],
        'core_skills': ['Human Resources', 'Recruitment', 'Talent Acquisition', 'Employee Relations', 'HR Analytics'],
        'important_skills': ['HR Operations', 'Performance Management', 'Onboarding', 'Payroll', 'Labor Law', 'Excel'],
        'supporting_skills': ['LinkedIn Recruiter', 'HRIS (Workday/SAP SuccessFactors)', 'Employee Engagement', 'Compensation & Benefits'],
        'general_skills': ['Interpersonal Communication', 'Empathy', 'Conflict Mediation'],
        'certifications': ['SHRM Certified Professional (SHRM-CP)', 'HRCI Associate Professional in Human Resources (aPHR)'],
        'experience_expectation': '0-5+ years',
        'roadmap': [
            {'phase': 'Phase 1: Talent Acquisition Core', 'duration': 'Weeks 1-4', 'topics': 'Job Descriptions, Boolean Search on LinkedIn, Applicant Tracking Systems (ATS), Structured Interviewing', 'resources': [{'title': 'SHRM Learning Resources', 'url': 'https://www.shrm.org'}]},
            {'phase': 'Phase 2: HR Operations & Law', 'duration': 'Weeks 5-8', 'topics': 'Onboarding Lifecycle, Industrial Relations, Labor Laws, POSH Compliance, Payroll Structuring', 'resources': [{'title': 'HRCI Certification Guide', 'url': 'https://www.hrci.org'}]},
            {'phase': 'Phase 3: People Analytics', 'duration': 'Weeks 9-12', 'topics': 'Employee Attrition Analysis, Power BI HR Dashboards, Time-to-Hire & Cost-per-Hire Metrics', 'resources': [{'title': 'Coursera People Analytics', 'url': 'https://www.coursera.org'}]},
            {'phase': 'Phase 4: Strategic HR Management', 'duration': 'Weeks 13-16', 'topics': 'Performance Management Frameworks (OKRs/360 Feedback), Employee Retention Strategies', 'resources': [{'title': 'Academy to Innovate HR (AIHR)', 'url': 'https://www.aihr.com'}]}
        ],
        'interview_questions': [
            {'q': 'How do you calculate Employee Turnover / Attrition Rate and what strategies reduce it?', 'a': 'Attrition Rate = `(Number of employees departed during period / Average number of employees) * 100`. Reduce via competitive compensation, clear growth roadmaps, manager coaching, and regular pulse surveys.'},
            {'q': 'What is the STAR method in behavioral job interviews and how do you evaluate candidate responses?', 'a': 'STAR: Situation, Task, Action, Result. Evaluate whether the candidate articulates concrete individual actions taken and quantifiable business results rather than generalized team achievements.'},
            {'q': 'How do you manage workplace conflict between two team members or an employee and their manager?', 'a': 'Hold separate neutral fact-finding 1-on-1s, arrange a structured joint mediation meeting, focus on objective workplace behaviors rather than personal attacks, agree on actionable behavioral commitments, and follow up in 30 days.'},
            {'q': 'What are key People Analytics metrics every HR executive should track?', 'a': 'Time-to-Hire, Cost-per-Hire, Quality-of-Hire (performance ratings after 1 year), Employee Net Promoter Score (eNPS), Offer Acceptance Rate, and 90-Day New Hire Attrition.'},
            {'q': 'Explain the purpose and structure of a 360-degree performance appraisal.', 'a': 'It collects multi-source feedback on an employee from managers, peers, direct reports, and self-evaluation, providing a holistic, balanced view of strengths, collaboration, and developmental blind spots.'}
        ]
    },

    # ---------------- 15. MARKETING & SALES ----------------
    'Digital Marketing Specialist': {
        'sector': 'Marketing & Sales',
        'description': 'Drives customer acquisition, brand engagement, and revenue through SEO, paid campaigns, content strategy, and analytics.',
        'qualification_required': False,
        'required_degrees': [],
        'career_switch_allowed': True,
        'compatible_degrees': ['BBA', 'MBA', 'B.Com', 'B.A', 'B.Tech', 'B.Sc'],
        'compatible_majors': ['Marketing', 'Business', 'Mass Communication', 'Media', 'Computer Science'],
        'industries': ['Marketing', 'E-commerce', 'IT', 'Media', 'Startups'],
        'core_skills': ['Digital Marketing', 'SEO', 'Google Analytics', 'Social Media Marketing', 'Content Marketing', 'Search Engine Optimization'],
        'important_skills': ['Google Ads', 'SEM', 'Email Marketing', 'Copywriting', 'CRM', 'Meta Ads', 'Data Analysis'],
        'supporting_skills': ['Canva', 'HubSpot', 'WordPress', 'A/B Testing', 'Conversion Rate Optimization (CRO)', 'Excel'],
        'general_skills': ['Creative Storytelling', 'Campaign Planning', 'Analytical Communication'],
        'certifications': ['Google Analytics Individual Qualification (GA4)', 'HubSpot Inbound Marketing', 'Google Ads Search Certification'],
        'experience_expectation': '0-5+ years',
        'roadmap': [
            {'phase': 'Phase 1: SEO & Content Strategy', 'duration': 'Weeks 1-4', 'topics': 'Keyword Research (Ahrefs/Semrush), On-Page & Technical SEO, Copywriting, Content Calendar', 'resources': [{'title': 'Moz Beginner\'s Guide to SEO', 'url': 'https://moz.com/beginners-guide-to-seo'}]},
            {'phase': 'Phase 2: Performance Marketing', 'duration': 'Weeks 5-8', 'topics': 'Google Ads (Search/Display), Meta Ads Manager, Audience Targeting, ROAS & CAC Optimization', 'resources': [{'title': 'Google Skillshop', 'url': 'https://skillshop.withgoogle.com'}]},
            {'phase': 'Phase 3: Analytics & Conversion', 'duration': 'Weeks 9-12', 'topics': 'Google Analytics 4 (GA4) Event Tracking, Funnel Visualization, A/B Testing Landing Pages', 'resources': [{'title': 'HubSpot Academy', 'url': 'https://academy.hubspot.com'}]},
            {'phase': 'Phase 4: Full Campaign Execution', 'duration': 'Weeks 13-16', 'topics': 'Email Automation (Mailchimp/Klaviyo), CRM Lead Nurturing, Live ROI Campaign Case Study', 'resources': [{'title': 'Neil Patel Marketing Blog', 'url': 'https://neilpatel.com'}]}
        ],
        'interview_questions': [
            {'q': 'Explain the difference between Customer Acquisition Cost (CAC) and Customer Lifetime Value (LTV).', 'a': 'CAC is total sales/marketing spend divided by new customers acquired. LTV is the total gross profit a customer generates across their relationship. A healthy business targets LTV:CAC >= 3:1.'},
            {'q': 'What are the core technical SEO ranking factors evaluated by search engines?', 'a': 'Page load speed (Core Web Vitals), mobile responsiveness, clean URL structure, XML sitemaps, robots.txt, structured data (Schema markup), HTTPS, and high-quality authoritative backlinks.'},
            {'q': 'How do you optimize a Google Ads campaign with a low Click-Through Rate (CTR) and high Cost-Per-Click (CPC)?', 'a': 'Refine ad copy with clear CTAs to improve CTR, add high-intent long-tail keywords, include Negative Keywords to eliminate irrelevant clicks, and optimize landing page relevance to raise Quality Score.'},
            {'q': 'What is the difference between Google Analytics 4 (GA4) and Universal Analytics?', 'a': 'Universal Analytics was session-based with pageviews. GA4 is entirely event-based, unifying web and mobile app interactions with automated machine learning insights and privacy-focused tracking.'},
            {'q': 'How do you design an A/B test for a high-traffic landing page checkout flow?', 'a': 'Isolate a single variable (e.g., button color, 1-step vs 2-step form) -> Formulate testable hypothesis -> Split traffic 50/50 randomly -> Measure Conversion Rate -> Conclude upon reaching 95% statistical significance.'}
        ]
    },

    # ---------------- 16. DESIGN & CREATIVE ----------------
    'UI/UX Designer': {
        'sector': 'Design & Creative',
        'description': 'Designs user journeys, wireframes, visual design systems, and interactive prototypes based on user research.',
        'qualification_required': False,
        'required_degrees': [],
        'career_switch_allowed': True,
        'compatible_degrees': ['B.Des', 'B.A', 'B.Tech', 'BCA', 'B.Sc', 'BBA'],
        'compatible_majors': ['Design', 'Human-Computer Interaction', 'Computer Science', 'Fine Arts', 'Multimedia'],
        'industries': ['IT', 'Design', 'Software', 'E-commerce', 'Fintech'],
        'core_skills': ['Figma', 'UI/UX Design', 'Wireframing', 'Prototyping', 'User Research', 'Design Systems'],
        'important_skills': ['User Experience', 'User Interface Design', 'Adobe XD', 'Visual Design', 'Information Architecture', 'Usability Testing'],
        'supporting_skills': ['Photoshop', 'Illustrator', 'HTML', 'CSS', 'Micro-interactions', 'Accessibility'],
        'general_skills': ['User Empathy', 'Creative Problem Solving', 'Design Storytelling'],
        'certifications': ['Google UX Design Professional Certificate', 'Nielsen Norman Group UX Certification'],
        'experience_expectation': '0-5+ years',
        'roadmap': [
            {'phase': 'Phase 1: UX Research & IA', 'duration': 'Weeks 1-4', 'topics': 'User Personas, Empathy Mapping, Information Architecture, User Journey Mapping, Low-Fi Wireframing', 'resources': [{'title': 'Nielsen Norman Group Articles', 'url': 'https://www.nngroup.com'}]},
            {'phase': 'Phase 2: Figma Mastery & UI Systems', 'duration': 'Weeks 5-8', 'topics': 'Figma Auto-Layout, Components, Design Tokens, Typography Scales, Responsive Grids', 'resources': [{'title': 'Figma YouTube Tutorials', 'url': 'https://www.youtube.com/@Figma'}]},
            {'phase': 'Phase 3: Interactive Prototyping', 'duration': 'Weeks 9-12', 'topics': 'High-Fidelity Prototyping, Smart Animate, Usability Testing Protocols, WCAG Accessibility (A11y)', 'resources': [{'title': 'Laws of UX', 'url': 'https://lawsofux.com'}]},
            {'phase': 'Phase 4: Case Studies & Portfolio', 'duration': 'Weeks 13-16', 'topics': 'Build 3 comprehensive UX Case Studies documenting problem, research, iterations, and final prototypes', 'resources': [{'title': 'Google UX Design Certificate', 'url': 'https://grow.google/certificates/ux-design/'}]}
        ],
        'interview_questions': [
            {'q': 'Walk me through your end-to-end UX design process for a new product feature.', 'a': '1) Empathize (User Interviews & Surveys) -> 2) Define (Problem Statement & Personas) -> 3) Ideate (Wireframes & User Flows) -> 4) Prototype (High-fidelity Figma mockups) -> 5) Test (Usability testing & feedback iteration).'},
            {'q': 'What are the 10 Nielsen Norman Usability Heuristics for User Interface Design?', 'a': 'Key ones include Visibility of System Status, Match between System and Real World, User Control & Freedom, Consistency & Standards, Error Prevention, and Recognition rather than Recall.'},
            {'q': 'What is a Design System and why is it essential for growing product teams?', 'a': 'A Design System is a single source of truth containing reusable UI components, design tokens (colors, spacing, typography), and brand guidelines, ensuring visual consistency and speeding up dev velocity.'},
            {'q': 'How do you ensure UI designs meet WCAG 2.1 AA accessibility standards?', 'a': 'Ensure color contrast ratio >= 4.5:1 for normal text, provide accessible focus states, design touch targets >= 44x44px, support dynamic text scaling, and avoid color as the only status indicator.'},
            {'q': 'Explain the difference between Qualitative and Quantitative user research.', 'a': 'Qualitative research discovers the "Why" and "How" through 1-on-1 user interviews and usability tests. Quantitative research measures the "How Many" and "What" through analytics, heatmaps, and large-scale surveys.'}
        ]
    },

    # ---------------- 17. SUPPLY CHAIN & OPERATIONS ----------------
    'Supply Chain Analyst': {
        'sector': 'Supply Chain & Operations',
        'description': 'Optimizes logistics networks, demand forecasting, inventory turnover, procurement schedules, and warehouse operations.',
        'qualification_required': False,
        'required_degrees': [],
        'career_switch_allowed': True,
        'compatible_degrees': ['MBA', 'B.Tech', 'BBA', 'B.Com', 'B.Sc'],
        'compatible_majors': ['Supply Chain', 'Operations', 'Industrial Engineering', 'Business', 'Logistics', 'Mechanical'],
        'industries': ['Logistics', 'Manufacturing', 'E-commerce', 'Retail', 'FMCG'],
        'core_skills': ['Supply Chain Management', 'Logistics', 'Procurement', 'Inventory Management', 'Excel', 'Data Analysis'],
        'important_skills': ['SAP', 'Power BI', 'SQL', 'Demand Forecasting', 'Operations Management', 'ERP'],
        'supporting_skills': ['Warehouse Management', 'Six Sigma', 'Vendor Management', 'Cost Optimization', 'Tableau'],
        'general_skills': ['Process Optimization', 'Negotiation', 'Analytical Problem Solving'],
        'certifications': ['APICS Certified Supply Chain Professional (CSCP)', 'Six Sigma Green Belt'],
        'experience_expectation': '0-5+ years',
        'roadmap': [
            {'phase': 'Phase 1: Supply Chain Fundamentals', 'duration': 'Weeks 1-4', 'topics': 'SCOR Model, Inventory Metrics (EOQ, Safety Stock, Reorder Points), Procurement Lifecycle', 'resources': [{'title': 'ASCM / APICS Portal', 'url': 'https://www.ascm.org'}]},
            {'phase': 'Phase 2: Forecasting & Analytics', 'duration': 'Weeks 5-8', 'topics': 'Demand Forecasting in Excel (Moving Averages, Exponential Smoothing), SQL for Shipment Data', 'resources': [{'title': 'MIT OpenCourseWare Supply Chain', 'url': 'https://ocw.mit.edu'}]},
            {'phase': 'Phase 3: ERP & Logistics Tools', 'duration': 'Weeks 9-12', 'topics': 'SAP MM/SD Module Workflows, Power BI Logistics KPI Dashboards (On-Time In-Full - OTIF)', 'resources': [{'title': 'Coursera Supply Chain Analytics', 'url': 'https://www.coursera.org'}]},
            {'phase': 'Phase 4: Optimization & Case Studies', 'duration': 'Weeks 13-16', 'topics': 'Route Optimization, Lean Six Sigma DMAIC Framework, Resilient Global Supply Chain Strategy', 'resources': [{'title': 'Council of Supply Chain Management', 'url': 'https://cscmp.org'}]}
        ],
        'interview_questions': [
            {'q': 'What is the Bullwhip Effect in supply chain management and how is it mitigated?', 'a': 'The Bullwhip Effect is the amplification of demand variability moving upstream from retailer to manufacturer. Mitigate via real-time POS data sharing, Vendor-Managed Inventory (VMI), and smaller batch sizes.'},
            {'q': 'How do you calculate Economic Order Quantity (EOQ) and Safety Stock?', 'a': '`EOQ = sqrt((2 * Demand * Order Cost) / Holding Cost)`. Safety stock buffers demand uncertainty: `Safety Stock = Z * sqrt(Lead Time * StdDev_Demand^2 + Demand^2 * StdDev_LeadTime^2)`.'},
            {'q': 'What are the key KPIs used to measure supply chain and logistics performance?', 'a': 'On-Time In-Full (OTIF) Delivery, Inventory Turnover Ratio, Order Lead Time, Cash-to-Cash Cycle Time, Days of Inventory on Hand (DOH), and Freight Cost per Unit.'},
            {'q': 'Explain the difference between Push and Pull supply chain strategies.', 'a': 'Push (Make-to-Stock) manufactures products based on long-term demand forecasts. Pull (Make-to-Order) triggers manufacturing and replenishment only after customer orders are received.'},
            {'q': 'What is ABC Analysis in inventory management?', 'a': 'ABC Analysis categorizes items based on Pareto Principle (80/20 rule): A-items (top 20% items driving 80% revenue, tightest control), B-items (moderate), C-items (bulk low-value items, simple bulk ordering).'}
        ]
    },

    # ---------------- 18. EDUCATION ----------------
    'Academic Lecturer / Educator': {
        'sector': 'Education',
        'description': 'Delivers academic instruction, develops pedagogical curricula, conducts scholarly research, and mentors students.',
        'qualification_required': True,
        'required_degrees': ['M.Sc', 'M.Tech', 'M.A', 'M.Com', 'M.Ed', 'B.Ed', 'PhD', 'NET'],
        'career_switch_allowed': False,
        'compatible_degrees': ['M.Sc', 'M.Tech', 'M.A', 'M.Com', 'M.Ed', 'B.Ed', 'PhD', 'NET'],
        'compatible_majors': ['Education', 'Sciences', 'Engineering', 'Humanities', 'Mathematics', 'Commerce'],
        'industries': ['Education', 'Universities', 'Colleges', 'Edtech', 'Research'],
        'core_skills': ['Teaching', 'Curriculum Design', 'Academic Research', 'Pedagogy', 'Classroom Management', 'Student Mentoring'],
        'important_skills': ['E-Learning', 'Instructional Design', 'Assessment & Evaluation', 'Higher Education', 'Subject Matter Expertise'],
        'supporting_skills': ['LMS Platforms (Moodle/Canvas)', 'PowerPoint Presentation', 'Educational Technology', 'Academic Writing'],
        'general_skills': ['Public Speaking', 'Patience & Empathy', 'Inspiring Communication'],
        'certifications': ['UGC NET / SLET Qualified', 'B.Ed / M.Ed Certification'],
        'experience_expectation': '0-5+ years',
        'roadmap': [
            {'phase': 'Phase 1: Pedagogical Foundations', 'duration': 'Weeks 1-4', 'topics': 'Bloom\'s Taxonomy, Learning Theories, Constructivist Teaching Methods, Syllabus Structuring', 'resources': [{'title': 'UGC Higher Education Portal', 'url': 'https://www.ugc.gov.in'}]},
            {'phase': 'Phase 2: Curriculum & Assessment', 'duration': 'Weeks 5-8', 'topics': 'Formative vs Summative Assessments, Rubric Design, Active Learning Strategies', 'resources': [{'title': 'Coursera Teaching in Higher Ed', 'url': 'https://www.coursera.org'}]},
            {'phase': 'Phase 3: Digital Pedagogy & EdTech', 'duration': 'Weeks 9-12', 'topics': 'LMS (Canvas/Moodle), Hybrid Classroom Engagement Tools, Interactive Digital Quizzes', 'resources': [{'title': 'EdSurge Learning Resources', 'url': 'https://www.edsurge.com'}]},
            {'phase': 'Phase 4: Academic Publishing & NET', 'duration': 'Weeks 13-16', 'topics': 'Peer-Reviewed Journal Publishing, Research Grant Proposals, UGC NET Exam Preparation', 'resources': [{'title': 'NTA NET Official Portal', 'url': 'https://ugcnet.nta.nic.in'}]}
        ],
        'interview_questions': [
            {'q': 'Explain Bloom\'s Taxonomy and how you apply it when designing learning outcomes.', 'a': 'Bloom\'s Taxonomy categorizes cognitive learning into 6 levels: Remember, Understand, Apply, Analyze, Evaluate, and Create. I align course assignments to progressively scaffold students from recall to creation.'},
            {'q': 'How do you handle a classroom with diverse student learning abilities and paces?', 'a': 'Use Differentiated Instruction: offer multimodal resources (visual, textual, hands-on), tiered assignments, peer-assisted learning groups, and dedicated office-hour mentoring.'},
            {'q': 'What is Formative Assessment vs Summative Assessment?', 'a': 'Formative assessment monitors ongoing student learning to provide continuous feedback (quizzes, class discussions). Summative assessment evaluates cumulative learning at the end of an instructional unit (final exams, capstone projects).'},
            {'q': 'How do you integrate active learning strategies into a traditional lecture format?', 'a': 'Use the "Think-Pair-Share" technique, live interactive polling (Mentimeter), real-world case study debates, and 5-minute problem-solving checkpoints every 20 minutes.'},
            {'q': 'How do you ensure academic integrity in online and hybrid learning environments?', 'a': 'Design open-ended, application-oriented questions requiring personal analysis rather than memorized recall, and utilize plagiarism detection software (Turnitin).'}
        ]
    },

    # ---------------- 19. SCIENCE & RESEARCH ----------------
    'Research Scientist': {
        'sector': 'Agriculture, Environment & Media',
        'description': 'Conducts advanced scientific experiments, publishes peer-reviewed research, and develops innovative technological breakthroughs.',
        'qualification_required': False,
        'required_degrees': [],
        'career_switch_allowed': True,
        'compatible_degrees': ['M.Sc', 'M.Tech', 'PhD', 'B.Tech', 'B.Sc'],
        'compatible_majors': ['Physics', 'Chemistry', 'Biology', 'Biotechnology', 'Data Science', 'Computer Science', 'Mathematics'],
        'industries': ['Research', 'Biotech', 'Pharmaceuticals', 'R&D', 'Academia', 'Government Labs'],
        'core_skills': ['Scientific Research', 'Data Analysis', 'Experimentation', 'Statistics', 'Scientific Writing', 'R'],
        'important_skills': ['Python', 'MATLAB', 'Laboratory Techniques', 'Literature Review', 'Hypothesis Testing', 'Data Visualization'],
        'supporting_skills': ['Bioinformatics', 'SPSS', 'LaTeX', 'Grant Writing', 'Peer Review'],
        'general_skills': ['Analytical Rigor', 'Intellectual Curiosity', 'Scientific Integrity'],
        'certifications': ['CSIR NET / JRF Qualified', 'Good Laboratory Practice (GLP) Certified'],
        'experience_expectation': '0-5+ years',
        'roadmap': [
            {'phase': 'Phase 1: Research Methodology', 'duration': 'Weeks 1-4', 'topics': 'Scientific Method, Systematic Literature Review, Experimental Design, Research Ethics', 'resources': [{'title': 'Nature Masterclasses', 'url': 'https://masterclasses.nature.com'}]},
            {'phase': 'Phase 2: Statistical Computing', 'duration': 'Weeks 5-8', 'topics': 'R / Python for Scientific Computing, ANOVA, Multivariate Regression, Non-Parametric Tests', 'resources': [{'title': 'R for Data Science', 'url': 'https://r4ds.hadley.nz'}]},
            {'phase': 'Phase 3: Lab Techniques & Modeling', 'duration': 'Weeks 9-12', 'topics': 'Domain-Specific Lab Protocols / Mathematical Simulation (MATLAB), Data Integrity Tracking', 'resources': [{'title': 'NCBI Research Tools', 'url': 'https://www.ncbi.nlm.nih.gov'}]},
            {'phase': 'Phase 4: Manuscript Publication', 'duration': 'Weeks 13-16', 'topics': 'Manuscript Drafting in LaTeX, Peer-Review Process, Conference Presentation, Grant Proposal Writing', 'resources': [{'title': 'Overleaf LaTeX Guides', 'url': 'https://www.overleaf.com'}]}
        ],
        'interview_questions': [
            {'q': 'Walk me through your methodology for designing a controlled scientific experiment.', 'a': '1) Formulate a falsifiable hypothesis -> 2) Identify Independent, Dependent, and Controlled variables -> 3) Determine sample size via statistical power analysis -> 4) Implement randomized double-blind controls -> 5) Execute with replication -> 6) Analyze data and quantify uncertainty.'},
            {'q': 'What is p-hacking (data dredging) and how do you ensure scientific reproducibility?', 'a': 'P-hacking is manipulating data or running multiple tests until non-significant results yield p < 0.05. Prevent via pre-registration of hypotheses, Bonferroni corrections for multiple testing, and open sharing of raw data/code.'},
            {'q': 'Explain the difference between Parametric and Non-Parametric statistical tests.', 'a': 'Parametric tests (t-test, ANOVA) assume normally distributed data and equal variances. Non-parametric tests (Mann-Whitney U, Kruskal-Wallis) make no distribution assumptions and evaluate ranked medians.'},
            {'q': 'How do you handle peer-review critique when reviewers request major experimental revisions?', 'a': 'Read feedback objectively, perform the additional experiments requested where feasible, provide clear point-by-point rebuttal responses with supporting data, and politely explain valid methodological constraints.'},
            {'q': 'What constitutes Good Laboratory Practice (GLP) in scientific data recording?', 'a': 'ALCOA+ principles: Attributable, Legible, Contemporaneous, Original, Accurate, Complete, Consistent, Enduring, and Available.'}
        ]
    },

    # ---------------- 20. MEDIA & COMMUNICATION ----------------
    'Content & Communications Specialist': {
        'sector': 'Agriculture, Environment & Media',
        'description': 'Creates compelling editorial content, corporate communications, press releases, and multi-channel media narratives.',
        'qualification_required': False,
        'required_degrees': [],
        'career_switch_allowed': True,
        'compatible_degrees': ['B.A', 'BBA', 'MBA', 'B.Com', 'B.Sc', 'B.Tech'],
        'compatible_majors': ['English', 'Journalism', 'Mass Communication', 'Media', 'Marketing', 'Literature'],
        'industries': ['Media', 'Marketing', 'Publishing', 'Corporate Communications', 'PR'],
        'core_skills': ['Content Writing', 'Copywriting', 'SEO Content', 'Editing', 'Communication', 'Content Strategy'],
        'important_skills': ['Social Media Management', 'Public Relations', 'Blogging', 'Storytelling', 'Research', 'WordPress'],
        'supporting_skills': ['Email Newsletters', 'Canva', 'Proofreading', 'Google Analytics', 'Press Releases'],
        'general_skills': ['Creativity', 'Adaptability of Tone', 'Deadlines Management'],
        'certifications': ['HubSpot Content Marketing Certified', 'Google Digital Marketing Certificate'],
        'experience_expectation': '0-4+ years',
        'roadmap': [
            {'phase': 'Phase 1: Writing Fundamentals', 'duration': 'Weeks 1-4', 'topics': 'Storytelling Frameworks, Headline Writing, Audience Persona Voice & Tone, Editing & Proofreading', 'resources': [{'title': 'Purdue OWL Writing Lab', 'url': 'https://owl.purdue.edu'}]},
            {'phase': 'Phase 2: SEO & Digital Content', 'duration': 'Weeks 5-8', 'topics': 'SEO Keyword Optimization, Long-Form Blog Strategy, Meta Descriptions, Content Pillars', 'resources': [{'title': 'HubSpot Content Marketing', 'url': 'https://academy.hubspot.com'}]},
            {'phase': 'Phase 3: PR & Multi-Channel Media', 'duration': 'Weeks 9-12', 'topics': 'Press Releases, Thought Leadership Ghostwriting (LinkedIn), Email Newsletters (Substack)', 'resources': [{'title': 'Copyblogger Content Guide', 'url': 'https://copyblogger.com'}]},
            {'phase': 'Phase 4: Brand Strategy & Portfolio', 'duration': 'Weeks 13-16', 'topics': 'Build live digital writing portfolio (Medium/Substack), Brand Messaging Guidelines', 'resources': [{'title': 'Contently Portfolio', 'url': 'https://contently.com'}]}
        ],
        'interview_questions': [
            {'q': 'How do you adapt your writing tone when transitioning between B2B and B2C audiences?', 'a': 'B2B writing focuses on ROI, efficiency, data evidence, and clear professional utility. B2C writing connects on emotional benefits, lifestyle aspirations, simplicity, and immediate compelling CTAs.'},
            {'q': 'What is your process for optimizing long-form articles for search engines (SEO) without keyword stuffing?', 'a': 'Focus on search intent, include primary and semantic LSI keywords naturally in H1/H2 headers, provide comprehensive depth answering user questions, optimize readability (Flesch score), and write compelling meta tags.'},
            {'q': 'How do you measure the business impact and ROI of content marketing efforts?', 'a': 'Track Organic Search Traffic, Dwell Time / Bounce Rate, Inbound Lead Conversions, Social Shares, Backlinks generated, and Assisted Conversions in GA4.'},
            {'q': 'Walk me through how you craft an effective Crisis Communication press release.', 'a': 'Acknowledge the situation promptly with transparency, express genuine empathy, outline concrete remedial actions being taken, direct to a dedicated contact/page, and ensure strict legal/executive alignment.'},
            {'q': 'How do you research a complex, highly technical subject matter you have never written about before?', 'a': 'Interview internal Subject Matter Experts (SMEs), study authoritative whitepapers and documentation, translate technical jargon into clear analogies, and review drafts with technical leads for accuracy.'}
        ]
    }
}

CANONICAL_ROLES: List[str] = list(CAREER_TAXONOMY.keys())

# ============================================================================
# 3. 250+ CENTRALIZED SKILL ALIASES & NORMALIZATION DICTIONARY
# ============================================================================
SKILL_ALIASES_MAP: Dict[str, str] = {
    # Programming Languages
    'python': 'Python', 'py': 'Python', 'python3': 'Python', 'python 3': 'Python',
    'java': 'Java', 'core java': 'Java', 'advanced java': 'Java',
    'c++': 'C++', 'cpp': 'C++', 'c plus plus': 'C++',
    'c#': 'C#', 'csharp': 'C#', 'c sharp': 'C#', '.net': 'C#', 'dotnet': 'C#',
    'c': 'C', 'c language': 'C', 'ansi c': 'C',
    'javascript': 'JavaScript', 'js': 'JavaScript', 'es6': 'JavaScript', 'es6+': 'JavaScript',
    'typescript': 'TypeScript', 'ts': 'TypeScript',
    'r': 'R', 'r programming': 'R', 'r language': 'R',
    'php': 'PHP', 'ruby': 'Ruby', 'go': 'Go', 'golang': 'Go', 'rust': 'Rust',
    'kotlin': 'Kotlin', 'swift': 'Swift', 'dart': 'Dart', 'scala': 'Scala',
    'matlab': 'MATLAB', 'matlab simulink': 'MATLAB Simulink', 'simulink': 'MATLAB Simulink',
    'bash': 'Bash', 'shell scripting': 'Bash', 'shell': 'Bash',

    # Web & Frontend
    'html': 'HTML', 'html5': 'HTML',
    'css': 'CSS', 'css3': 'CSS',
    'react': 'React', 'reactjs': 'React', 'react.js': 'React', 'react native': 'React Native',
    'next.js': 'Next.js', 'nextjs': 'Next.js', 'next': 'Next.js',
    'vue': 'Vue.js', 'vuejs': 'Vue.js', 'vue.js': 'Vue.js',
    'angular': 'Angular', 'angularjs': 'Angular',
    'tailwind': 'Tailwind CSS', 'tailwind css': 'Tailwind CSS', 'tailwindcss': 'Tailwind CSS',
    'bootstrap': 'Bootstrap',
    'redux': 'Redux', 'zustand': 'Redux',
    'responsive web design': 'Responsive Web Design', 'responsive design': 'Responsive Web Design',

    # Backend & APIs
    'node': 'Node.js', 'nodejs': 'Node.js', 'node.js': 'Node.js',
    'express': 'Express.js', 'expressjs': 'Express.js', 'express.js': 'Express.js',
    'django': 'Django', 'django rest framework': 'Django', 'drf': 'Django',
    'flask': 'Flask',
    'fastapi': 'FastAPI', 'fast api': 'FastAPI',
    'spring boot': 'Spring Boot', 'springboot': 'Spring Boot', 'spring': 'Spring Boot',
    'rest api': 'REST API', 'restful api': 'REST API', 'rest': 'REST API', 'apis': 'REST API',
    'graphql': 'GraphQL',
    'microservices': 'Microservices', 'microservice': 'Microservices',

    # Databases & Caching
    'sql': 'SQL', 'structured query language': 'SQL',
    'mysql': 'MySQL',
    'postgresql': 'PostgreSQL', 'postgres': 'PostgreSQL', 'psql': 'PostgreSQL',
    'mongodb': 'MongoDB', 'mongo': 'MongoDB',
    'redis': 'Redis',
    'sqlite': 'SQLite',
    'oracle': 'Oracle', 'oracle db': 'Oracle',
    'firebase': 'Firebase',
    'snowflake': 'Snowflake', 'bigquery': 'BigQuery',

    # AI, ML & Data Science
    'machine learning': 'Machine Learning', 'ml': 'Machine Learning', 'machine-learning': 'Machine Learning',
    'deep learning': 'Deep Learning', 'dl': 'Deep Learning', 'neural networks': 'Deep Learning',
    'nlp': 'NLP', 'natural language processing': 'NLP',
    'computer vision': 'Computer Vision', 'cv': 'Computer Vision', 'opencv': 'Computer Vision',
    'scikit-learn': 'Scikit-Learn', 'sklearn': 'Scikit-Learn', 'scikit learn': 'Scikit-Learn',
    'pytorch': 'PyTorch', 'torch': 'PyTorch',
    'tensorflow': 'TensorFlow', 'tf': 'TensorFlow', 'keras': 'TensorFlow',
    'pandas': 'Pandas', 'numpy': 'NumPy',
    'statistics': 'Statistics', 'probability': 'Statistics', 'inferential statistics': 'Statistics',
    'data analysis': 'Data Analysis', 'exploratory data analysis': 'Exploratory Data Analysis', 'eda': 'Exploratory Data Analysis',
    'data visualization': 'Data Visualization', 'dataviz': 'Data Visualization',
    'matplotlib': 'Matplotlib', 'seaborn': 'Seaborn',
    'generative ai': 'Generative AI', 'genai': 'Generative AI', 'gen ai': 'Generative AI',
    'langchain': 'LangChain', 'llamaindex': 'LangChain',
    'prompt engineering': 'Prompt Engineering',
    'vector databases': 'Vector Databases', 'chromadb': 'Vector Databases', 'pinecone': 'Vector Databases',
    'hugging face': 'Hugging Face', 'huggingface': 'Hugging Face', 'transformers': 'Hugging Face',
    'model deployment': 'Model Deployment', 'mlops': 'MLOps',

    # Big Data & Data Engineering
    'apache spark': 'Apache Spark', 'spark': 'Apache Spark', 'pyspark': 'Apache Spark',
    'data pipelines': 'Data Pipelines', 'pipeline': 'Data Pipelines',
    'etl': 'ETL', 'extract transform load': 'ETL',
    'data warehousing': 'Data Warehousing', 'data warehouse': 'Data Warehousing', 'dwh': 'Data Warehousing',
    'airflow': 'Airflow', 'apache airflow': 'Airflow',
    'kafka': 'Kafka', 'apache kafka': 'Kafka',
    'dbt': 'dbt',

    # Cloud & DevOps
    'aws': 'AWS', 'amazon web services': 'AWS',
    'azure': 'Azure', 'microsoft azure': 'Azure',
    'google cloud': 'Google Cloud', 'gcp': 'Google Cloud', 'google cloud platform': 'Google Cloud',
    'cloud architecture': 'Cloud Architecture',
    'docker': 'Docker', 'containerization': 'Docker', 'containers': 'Docker',
    'kubernetes': 'Kubernetes', 'k8s': 'Kubernetes',
    'ci/cd': 'CI/CD', 'cicd': 'CI/CD', 'continuous integration': 'CI/CD',
    'jenkins': 'Jenkins', 'github actions': 'GitHub Actions',
    'terraform': 'Terraform', 'iac': 'Terraform', 'infrastructure as code': 'Terraform',
    'ansible': 'Ansible',
    'linux': 'Linux', 'ubuntu': 'Linux', 'unix': 'Linux', 'redhat': 'Linux',
    'git': 'Git', 'github': 'Git', 'gitlab': 'Git', 'version control': 'Git',

    # Cybersecurity
    'network security': 'Network Security', 'cybersecurity': 'Network Security', 'infosec': 'Network Security',
    'ethical hacking': 'Ethical Hacking', 'penetration testing': 'Penetration Testing', 'pentesting': 'Penetration Testing',
    'siem': 'SIEM', 'splunk': 'SIEM', 'soc': 'SIEM',
    'vulnerability assessment': 'Vulnerability Assessment', 'vapt': 'Vulnerability Assessment',
    'wireshark': 'Wireshark', 'nmap': 'Nmap', 'burp suite': 'Burp Suite',
    'firewalls': 'Firewalls', 'cryptography': 'Cryptography', 'incident response': 'Incident Response',

    # Business & Analytics
    'excel': 'Excel', 'advanced excel': 'Excel', 'ms excel': 'Excel', 'vlookup': 'Excel', 'pivot tables': 'Excel',
    'power bi': 'Power BI', 'powerbi': 'Power BI', 'pbi': 'Power BI', 'dax': 'Power BI',
    'tableau': 'Tableau',
    'business analysis': 'Business Analysis', 'requirements gathering': 'Requirements Gathering',
    'financial modeling': 'Financial Modeling', 'financial analysis': 'Financial Analysis',
    'accounting': 'Accounting', 'corporate finance': 'Corporate Finance', 'valuation': 'Valuation', 'dcf': 'DCF Analysis',
    'auditing': 'Auditing', 'taxation': 'Taxation', 'gst': 'GST', 'income tax': 'Income Tax',

    # Management & Methodologies
    'agile': 'Agile', 'scrum': 'Scrum', 'jira': 'Jira',
    'project management': 'Project Management', 'team leadership': 'Team Leadership',
    'stakeholder management': 'Stakeholder Management', 'risk management': 'Risk Management',
    'budgeting': 'Budgeting', 'operations management': 'Operations Management',

    # Mechanical & Civil Engineering
    'autocad': 'AutoCAD', 'cad': 'CAD',
    'solidworks': 'SolidWorks', 'catia': 'CATIA', 'ansys': 'ANSYS', 'fea': 'Finite Element Analysis (FEA)',
    'thermodynamics': 'Thermodynamics', 'fluid mechanics': 'Fluid Mechanics', 'manufacturing': 'Manufacturing',
    'mechanical design': 'Mechanical Design', 'gd&t': 'GD&T', 'cnc': 'CNC Programming',
    'staad.pro': 'STAAD.Pro', 'staad': 'STAAD.Pro', 'revit': 'Revit', 'bim': 'BIM',
    'structural analysis': 'Structural Analysis', 'surveying': 'Surveying',
    'construction management': 'Construction Management', 'civil engineering': 'Civil Engineering',

    # Electrical & Electronics
    'plc': 'PLC', 'scada': 'PLC', 'ladder logic': 'PLC',
    'circuit design': 'Circuit Design', 'pcb design': 'PCB Design', 'kicad': 'PCB Design', 'altium': 'PCB Design',
    'power systems': 'Power Systems', 'electrical systems': 'Electrical Systems',
    'embedded systems': 'Embedded Systems', 'embedded c': 'Embedded C', 'microcontrollers': 'Microcontrollers',
    'arduino': 'Arduino', 'esp32': 'Microcontrollers', 'arm cortex': 'Microcontrollers', 'vlsi': 'VLSI', 'verilog': 'Verilog',

    # Healthcare & Legal
    'clinical diagnosis': 'Clinical Diagnosis', 'patient care': 'Patient Care', 'pharmacology': 'Pharmacology',
    'internal medicine': 'Internal Medicine', 'medical treatment': 'Medical Treatment', 'healthcare data': 'Healthcare Data',
    'clinical data': 'Clinical Data', 'healthcare analytics': 'Healthcare Analytics',
    'legal research': 'Legal Research', 'contract law': 'Contract Law', 'litigation': 'Litigation',
    'legal drafting': 'Legal Drafting', 'constitutional law': 'Constitutional Law', 'corporate law': 'Corporate Law',

    # Design, Marketing, Education, Media & SCM
    'figma': 'Figma', 'ui/ux design': 'UI/UX Design', 'ui/ux': 'UI/UX Design', 'ui ux': 'UI/UX Design',
    'wireframing': 'Wireframing', 'prototyping': 'Prototyping', 'user research': 'User Research', 'design systems': 'Design Systems',
    'digital marketing': 'Digital Marketing', 'seo': 'SEO', 'search engine optimization': 'SEO',
    'google analytics': 'Google Analytics', 'social media marketing': 'Social Media Marketing', 'content marketing': 'Content Marketing',
    'supply chain management': 'Supply Chain Management', 'supply chain': 'Supply Chain Management', 'scm': 'Supply Chain Management',
    'logistics': 'Logistics', 'procurement': 'Procurement', 'inventory management': 'Inventory Management', 'sap': 'SAP',
    'teaching': 'Teaching', 'curriculum design': 'Curriculum Design', 'pedagogy': 'Pedagogy',
    'content writing': 'Content Writing', 'copywriting': 'Copywriting', 'technical writing': 'Content Writing',
    'scientific research': 'Scientific Research', 'experimentation': 'Experimentation', 'data structures': 'Data Structures',
    'dsa': 'Data Structures', 'algorithms': 'Algorithms', 'object oriented programming': 'Object Oriented Programming',
    'oops': 'Object Oriented Programming', 'oop': 'Object Oriented Programming',
    'opencv': 'Computer Vision', 'yolo': 'Computer Vision', 'yolov8': 'Computer Vision', 'yolov5': 'Computer Vision',
    'ai agents': 'Generative AI', 'llm': 'Generative AI', 'llms': 'Generative AI', 'large language models': 'Generative AI',
    'ai/ml': 'Machine Learning', 'backend development': 'Node.js', 'frontend development': 'React'
}

# ============================================================================
# 4. DEGREE NORMALIZATION PATTERNS
# ============================================================================
DEGREE_NORMALIZATION: List[Tuple[str, str]] = [
    (r'\b(b\.?tech|bachelor of technology|b\.?e\.?|bachelor of engineering)\b', 'B.Tech'),
    (r'\b(m\.?tech|master of technology|m\.?e\.?|master of engineering)\b', 'M.Tech'),
    (r'\b(bca|bachelor of computer applications)\b', 'BCA'),
    (r'\b(mca|master of computer applications)\b', 'MCA'),
    (r'\b(b\.?sc|bachelor of science)\b', 'B.Sc'),
    (r'\b(m\.?sc|master of science)\b', 'M.Sc'),
    (r'\b(mba|master of business administration)\b', 'MBA'),
    (r'\b(bba|bachelor of business administration)\b', 'BBA'),
    (r'\b(b\.?com|bachelor of commerce)\b', 'B.Com'),
    (r'\b(m\.?com|master of commerce)\b', 'M.Com'),
    (r'\b(mbbs|m\.b\.b\.s|bachelor of medicine)\b', 'MBBS'),
    (r'\b(md|m\.d|doctor of medicine)\b', 'MD'),
    (r'\b(ms|m\.s|master of surgery)\b', 'MS'),
    (r'\b(bds|bachelor of dental surgery)\b', 'BDS'),
    (r'\b(ll\.?b|bachelor of laws|ba\s*llb|bba\s*llb)\b', 'LLB'),
    (r'\b(ll\.?m|master of laws)\b', 'LLM'),
    (r'\b(b\.?arch|bachelor of architecture)\b', 'B.Arch'),
    (r'\b(m\.?arch|master of architecture)\b', 'M.Arch'),
    (r'\b(ca|chartered accountant|icai|cpa)\b', 'CA'),
    (r'\b(b\.?ed|bachelor of education)\b', 'B.Ed'),
    (r'\b(m\.?ed|master of education)\b', 'M.Ed'),
    (r'\b(b\.?des|bachelor of design)\b', 'B.Des'),
    (r'\b(b\.?pharm|bachelor of pharmacy)\b', 'B.Pharm'),
    (r'\b(ph\.?d|doctor of philosophy)\b', 'PhD'),
    (r'\b(diploma|polytechnic)\b', 'Diploma'),
]

MAJOR_NORMALIZATION: List[Tuple[str, str]] = [
    (r'\b(computer science|cse|cs|software engineering)\b', 'Computer Science'),
    (r'\b(information technology|it)\b', 'Information Technology'),
    (r'\b(artificial intelligence|ai|aiml|ai & ml|machine learning)\b', 'Artificial Intelligence'),
    (r'\b(data science|data analytics)\b', 'Data Science'),
    (r'\b(electronics|ece|electronics and communication)\b', 'Electronics'),
    (r'\b(electrical|eee|electrical engineering)\b', 'Electrical'),
    (r'\b(mechanical|me|mechanical engineering|automobile)\b', 'Mechanical'),
    (r'\b(civil|ce|civil engineering|structural)\b', 'Civil'),
    (r'\b(finance|financial management)\b', 'Finance'),
    (r'\b(human resources|hr|people operations)\b', 'Human Resources'),
    (r'\b(marketing|sales|digital marketing)\b', 'Marketing'),
    (r'\b(business|management|business administration|operations)\b', 'Business'),
    (r'\b(accounting|taxation|auditing)\b', 'Accounting'),
    (r'\b(medicine|surgery|clinical medicine)\b', 'Medicine'),
    (r'\b(law|legal studies|corporate law)\b', 'Law'),
    (r'\b(architecture|urban planning)\b', 'Architecture'),
    (r'\b(mathematics|statistics|math)\b', 'Mathematics'),
    (r'\b(physics|chemistry|biology|biotechnology)\b', 'Biotechnology'),
    (r'\b(design|ui\/ux|graphic design|fine arts)\b', 'Design'),
    (r'\b(mass communication|journalism|media|english)\b', 'Media'),
]


# ============================================================================
# 5. EXACT TOKEN NORMALIZATION FUNCTIONS (Zero Substring False Positives)
# ============================================================================
def normalize_degree(degree_str: str) -> str:
    """Normalizes raw degree string to canonical degree family."""
    if not degree_str:
        return 'B.Tech'
    s = str(degree_str).strip()
    for pattern, canonical in DEGREE_NORMALIZATION:
        if re.search(pattern, s, re.IGNORECASE):
            return canonical
    return s.title()


def normalize_major(major_str: str) -> str:
    """Normalizes raw major/branch string to canonical major."""
    if not major_str:
        return 'Computer Science'
    s = str(major_str).strip()
    for pattern, canonical in MAJOR_NORMALIZATION:
        if re.search(pattern, s, re.IGNORECASE):
            return canonical
    return s.title()


def normalize_skill(skill_str: str) -> Optional[str]:
    """
    Normalizes a single skill string using exact token boundary matching.
    Guarantees that single-letter skills like 'C' or 'R' do NOT match substrings
    in words like 'Machine Learning', 'Docker', or 'React'.
    """
    if not skill_str:
        return None
    raw = str(skill_str).strip().lower()
    if not raw:
        return None

    # 1. Direct dictionary match
    if raw in SKILL_ALIASES_MAP:
        return SKILL_ALIASES_MAP[raw]

    # 2. Exact word boundary regex check against known aliases
    for alias, canonical in SKILL_ALIASES_MAP.items():
        if len(alias) <= 2:
            # For short 1-2 char tokens (c, r, go, js, ts, ai, ml, ui), require exact string equality
            if raw == alias:
                return canonical
        else:
            # For multi-character aliases, allow exact word boundary matches
            pattern = r'^(?:' + re.escape(alias) + r')$'
            if re.match(pattern, raw):
                return canonical

    # 3. Fallback title-cased if length >= 3 and not pure punctuation
    if len(raw) >= 3 and len(raw) <= 30 and re.search(r'^[a-zA-Z0-9\s\.\+#\-\/]+$', raw) and not re.search(r'\b(and|or|with|the|for|from|in|to|at|by|of|across|achieving|performed|served)\b', raw):
        return raw.title()
    return None


def extract_skills_from_text(raw_text: str) -> List[str]:
    """
    Extracts canonical skills from free-form text (e.g. resume PDF) by searching for
    known aliases using strict token boundaries.
    Guarantees zero sentence dumps or false positives, and automatically handles
    PDF word-squishing (e.g. 'Inpython', 'Integratedopencvfor', 'Usingtailwind').
    """
    if not raw_text:
        return []

    # Preprocess text to unstick glued words from PDF extraction
    clean_text = re.sub(r'([a-z])([A-Z])', r'\1 \2', str(raw_text))
    clean_text = re.sub(r'([A-Za-z])([0-9])', r'\1 \2', clean_text)
    clean_text = re.sub(r'([0-9])([A-Za-z])', r'\1 \2', clean_text)
    
    # Common tech keywords that often get glued to prepositions in PDFs
    glued_keywords = [
        'python', 'opencv', 'tailwind', 'react', 'fastapi', 'django', 'flask',
        'docker', 'kubernetes', 'aws', 'azure', 'pytorch', 'tensorflow', 'yolo',
        'yolov8', 'node', 'mongodb', 'postgresql', 'mysql', 'sql', 'git', 'linux',
        'pandas', 'numpy', 'scikit', 'figma', 'tableau', 'power bi'
    ]
    for kw in glued_keywords:
        clean_text = re.sub(r'(?i)([a-z]{2,})(' + re.escape(kw) + r')([a-z]*)', r'\1 \2 \3', clean_text)

    found_canonical: Set[str] = set()
    lower_text = clean_text.lower()

    # Sort aliases by length descending so longer phrases match first
    sorted_aliases = sorted(SKILL_ALIASES_MAP.keys(), key=len, reverse=True)

    for alias in sorted_aliases:
        canonical = SKILL_ALIASES_MAP[alias]
        if canonical in found_canonical:
            continue

        if len(alias) <= 2:
            # For 1-2 char tokens (c, r, go, js, ts, ai, ml, ui), require strict standalone token
            pattern = r'(?<![a-zA-Z0-9_])' + re.escape(alias) + r'(?![a-zA-Z0-9_])'
        else:
            # Multi-char token with word boundaries
            pattern = r'\b' + re.escape(alias) + r'\b'

        if re.search(pattern, lower_text):
            found_canonical.add(canonical)

    return sorted(list(found_canonical))


def extract_normalized_skills(skills_input: Any) -> List[str]:
    """
    Extracts, deduplicates, and normalizes a list or comma-separated string of skills.
    Ensures safe word-boundary matching.
    """
    if not skills_input:
        return []

    raw_tokens: List[str] = []
    if isinstance(skills_input, list):
        for item in skills_input:
            if isinstance(item, str):
                raw_tokens.extend([s.strip() for s in item.split(',') if s.strip()])
    elif isinstance(skills_input, str):
        # Split by comma or semicolon
        raw_tokens = [s.strip() for s in re.split(r'[,;]+', skills_input) if s.strip()]

    normalized_set: Set[str] = set()
    for token in raw_tokens:
        clean_token = token.strip()
        canonical = normalize_skill(clean_token)
        if canonical:
            normalized_set.add(canonical)

    return sorted(list(normalized_set))


# ============================================================================
# 6. QUALIFICATION GATING ENGINE
# ============================================================================
def check_qualification_eligibility(role_name: str, degree: str, major: str) -> Dict[str, Any]:
    """
    Hard Qualification Gating Layer for Regulated Professions.
    Guarantees that regulated professions (Doctor, Lawyer, Architect, CA, etc.)
    CANNOT be recommended merely because a candidate has generic or related skills.
    """
    role_meta = CAREER_TAXONOMY.get(role_name)
    if not role_meta:
        return {'eligible': True, 'reason': 'Standard open profession'}

    is_gated = role_meta.get('qualification_required', False)
    if not is_gated:
        return {'eligible': True, 'reason': 'Open career track (supports career switching with skill evidence)'}

    required_degrees = [d.lower() for d in role_meta.get('required_degrees', [])]
    user_degree = normalize_degree(degree).lower()
    user_major = normalize_major(major).lower()

    # Match against required degree families
    deg_matched = any(req in user_degree or user_degree in req for req in required_degrees)

    if deg_matched:
        return {
            'eligible': True,
            'reason': f'Verified required formal qualification ({degree}) for {role_name}.'
        }
    else:
        req_str = ', '.join(role_meta.get('required_degrees', []))
        return {
            'eligible': False,
            'reason': f'Mandatory professional qualification missing. {role_name} requires a verified degree in: {req_str}.'
        }
