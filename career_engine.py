# career_engine.py
"""
Career Engine for Edu2Job:
- Role Taxonomy for all 15 Canonical Roles
- Hybrid Career Match Scoring (ML Probability + Skill Fit + Academic Fit + Experience Fit + Industry Fit)
- Match Tier Classification
- Skill Gap Analysis, Step-by-Step Learning Roadmaps, and AI Mock Interview Questions
"""

import numpy as np
from typing import Dict, Any, List

CANONICAL_ROLES: List[str] = [
    'Data Scientist', 'Machine Learning Engineer', 'Data Analyst',
    'Software Engineer', 'Frontend Developer', 'Backend Developer',
    'Full Stack Developer', 'DevOps Engineer', 'Cloud Engineer',
    'Project Manager', 'Business Analyst', 'Financial Analyst',
    'Electrical Engineer', 'Mechanical Engineer', 'Civil Engineer'
]

HYBRID_WEIGHTS: Dict[str, float] = {
    'ml_probability': 0.35,
    'skill_fit': 0.40,
    'academic_fit': 0.15,
    'experience_fit': 0.05,
    'industry_fit': 0.05
}

ROLE_TAXONOMY: Dict[str, Dict[str, Any]] = {
    'Data Scientist': {
        'category': 'Data Science and AI',
        'degrees': ['B.Tech', 'M.Tech', 'M.Sc', 'B.Sc', 'PhD', 'MBA'],
        'majors': ['Computer Science', 'Data Science', 'Artificial Intelligence', 'Mathematics', 'Information Technology'],
        'industries': ['Data Science', 'IT', 'Finance', 'Healthcare'],
        'core_skills': ['Python', 'SQL', 'Machine Learning', 'Pandas', 'NumPy', 'Statistics'],
        'advanced_skills': ['Deep Learning', 'PyTorch', 'TensorFlow', 'Scikit-Learn', 'NLP', 'Computer Vision', 'Docker'],
        'roadmap': [
            {'phase': 'Phase 1: Foundations', 'duration': 'Weeks 1-4', 'topics': 'Python for Data Science, Advanced SQL, Statistics and Probability', 'resources': [{'title': 'Kaggle Python Course', 'url': 'https://www.kaggle.com/learn'}, {'title': 'Khan Academy Statistics', 'url': 'https://www.khanacademy.org/math/statistics-probability'}]},
            {'phase': 'Phase 2: Machine Learning', 'duration': 'Weeks 5-10', 'topics': 'Supervised and Unsupervised ML, Scikit-Learn, Feature Engineering, Model Evaluation', 'resources': [{'title': 'Scikit-Learn Tutorials', 'url': 'https://scikit-learn.org/stable/tutorial/index.html'}, {'title': 'StatQuest with Josh Starmer', 'url': 'https://www.youtube.com/@statquest'}]},
            {'phase': 'Phase 3: Production and Deep Learning', 'duration': 'Weeks 11-16', 'topics': 'PyTorch, FastAPI Model Deployment, Docker, End-to-End ML Projects', 'resources': [{'title': 'Fast.ai Deep Learning', 'url': 'https://course.fast.ai'}, {'title': 'Roadmap.sh AI Guide', 'url': 'https://roadmap.sh/ai-data-scientist'}]}
        ],
        'interview_questions': [
            {'q': 'Explain the difference between Overfitting and Underfitting, and how to prevent them?', 'a': 'Overfitting occurs when a model learns noise in training data (high variance). Prevent via regularization (L1/L2), cross-validation, dropout, and pruning. Underfitting is high bias (model too simple); solve by adding complexity or better features.'},
            {'q': 'What is the difference between Precision and Recall? When do you prioritize which?', 'a': 'Precision = TP/(TP+FP) measures accuracy of positive predictions. Recall = TP/(TP+FN) measures how many actual positives were caught. In spam detection prioritize Precision; in medical diagnostics prioritize Recall.'},
            {'q': 'Explain how Gradient Descent works in optimization.', 'a': 'Gradient descent iteratively calculates the partial derivatives of the loss function with respect to weights, taking steps proportional to the negative gradient scaled by the learning rate to find the global minimum.'},
            {'q': 'How do you handle imbalanced datasets in machine learning?', 'a': 'Use techniques like SMOTE (oversampling minority class), Random Undersampling, Cost-sensitive learning (class_weight), and evaluation metrics like PR-AUC / F1-Score instead of plain accuracy.'},
            {'q': 'Walk me through how Random Forest reduces variance compared to a single Decision Tree.', 'a': 'Random Forest builds multiple decision trees using Bagging (Bootstrap Aggregating) and random feature subspace selection. Averaging their predictions cancels out uncorrelated individual errors and slashes variance.'}
        ]
    },
    'Machine Learning Engineer': {
        'category': 'Data Science and AI',
        'degrees': ['B.Tech', 'M.Tech', 'MCA', 'B.Sc', 'PhD'],
        'majors': ['Computer Science', 'Artificial Intelligence', 'Data Science', 'Electronics'],
        'industries': ['IT', 'Data Science', 'Engineering'],
        'core_skills': ['Python', 'Machine Learning', 'Deep Learning', 'PyTorch', 'TensorFlow', 'Docker'],
        'advanced_skills': ['C++', 'NLP', 'Computer Vision', 'LangChain', 'FastAPI', 'Scikit-Learn', 'Git', 'MLOps'],
        'roadmap': [
            {'phase': 'Phase 1: ML & Math Foundations', 'duration': 'Weeks 1-4', 'topics': 'Linear Algebra, Calculus, Python OOP, Scikit-Learn, PyTorch Tensors', 'resources': [{'title': 'Deep Learning Specialization (Coursera)', 'url': 'https://www.deeplearning.ai'}, {'title': 'PyTorch 60min Blitz', 'url': 'https://pytorch.org/tutorials/beginner/blitz/'}]},
            {'phase': 'Phase 2: Deep Learning & Architectures', 'duration': 'Weeks 5-10', 'topics': 'CNNs, Transformers, Attention Mechanisms, Hugging Face Transformers, Transfer Learning', 'resources': [{'title': 'Hugging Face NLP Course', 'url': 'https://huggingface.co/learn/nlp-course'}, {'title': 'Stanford CS231n Computer Vision', 'url': 'https://cs231n.stanford.edu'}]},
            {'phase': 'Phase 3: MLOps & Production Pipelines', 'duration': 'Weeks 11-16', 'topics': 'Docker, MLflow, Triton Inference Server, ONNX Runtime, Model Monitoring', 'resources': [{'title': 'Made With ML (MLOps Guide)', 'url': 'https://madewithml.com'}, {'title': 'Full Stack Deep Learning', 'url': 'https://fullstackdeeplearning.com'}]}
        ],
        'interview_questions': [
            {'q': 'How does Self-Attention work in the Transformer architecture?', 'a': 'Self-Attention computes Queries (Q), Keys (K), and Values (V). Attention weights are calculated as softmax((Q @ K^T) / sqrt(d_k)) and multiplied by V, enabling direct tokens relation over long distances.'},
            {'q': 'What is the vanishing/exploding gradient problem and how is it resolved?', 'a': 'In deep networks, backpropagating gradients can shrink exponentially or explode. Mitigations include Residual Connections (ResNets), Batch/Layer Normalization, ReLU activations, and Xavier/He initialization.'},
            {'q': 'How do you optimize LLM / Deep Learning models for real-time low-latency inference?', 'a': 'Employ Quantization (INT8/INT4/FP8), Model Pruning, Knowledge Distillation, TensorRT/vLLM batching, KV caching, and ONNX Runtime graph optimizations.'},
            {'q': 'Explain the difference between Data Parallelism and Pipeline Parallelism in distributed training.', 'a': 'Data Parallelism replicates the entire model across GPUs with partitioned data batches and synchronized gradients. Pipeline Parallelism partitions layers across GPUs in a sequential pipeline.'},
            {'q': 'What is Catastrophic Forgetting in neural networks and how is it mitigated?', 'a': 'When fine-tuning on a new domain, weights overwrite previous knowledge. Mitigate via LoRA (Low-Rank Adaptation), Elastic Weight Consolidation (EWC), or replay buffers.'}
        ]
    },
    'Data Analyst': {
        'category': 'Analytics and BI',
        'degrees': ['B.Tech', 'BCA', 'B.Sc', 'MBA', 'BBA', 'B.Com', 'M.Sc'],
        'majors': ['Information Technology', 'Computer Science', 'Finance', 'Business', 'Mathematics'],
        'industries': ['Finance', 'IT', 'Consulting', 'Marketing', 'Healthcare'],
        'core_skills': ['SQL', 'Excel', 'Power BI', 'Python', 'Tableau'],
        'advanced_skills': ['Data Visualization', 'Data Analysis', 'Statistics', 'Pandas', 'Business Analysis', 'MySQL', 'A/B Testing'],
        'roadmap': [
            {'phase': 'Phase 1: Advanced Excel and SQL', 'duration': 'Weeks 1-4', 'topics': 'Pivot Tables, VLOOKUP/XLOOKUP, SQL Joins, Window Functions, Group By and Aggregations', 'resources': [{'title': 'SQLZoo Interactive SQL', 'url': 'https://sqlzoo.net'}, {'title': 'Chandoo Excel Analytics', 'url': 'https://chandoo.org'}]},
            {'phase': 'Phase 2: BI Tools and Dashboards', 'duration': 'Weeks 5-8', 'topics': 'Power BI (DAX, Data Modeling, Interactive Dashboards), Tableau Storytelling, KPI tracking', 'resources': [{'title': 'Microsoft Power BI Guided Learning', 'url': 'https://learn.microsoft.com/en-us/power-bi/'}, {'title': 'Tableau Public Visual Gallery', 'url': 'https://public.tableau.com'}]},
            {'phase': 'Phase 3: Python for Analytics', 'duration': 'Weeks 9-14', 'topics': 'Pandas and Seaborn for Exploratory Data Analysis, Cohort Analysis, A/B Testing, Executive Presentations', 'resources': [{'title': 'Kaggle Data Visualization Course', 'url': 'https://www.kaggle.com/learn/data-visualization'}, {'title': 'Google Data Analytics Certificate', 'url': 'https://grow.google/certificates/data-analytics/'}]}
        ],
        'interview_questions': [
            {'q': 'Explain the difference between WHERE and HAVING clauses in SQL.', 'a': 'WHERE filters individual rows before any aggregations are computed. HAVING filters aggregated groups after the GROUP BY clause has been applied.'},
            {'q': 'What is a SQL Window Function and how does it differ from GROUP BY?', 'a': 'Window functions (e.g. ROW_NUMBER, RANK, LAG/LEAD) perform calculations across a set of table rows related to the current row without collapsing rows into a single summary output.'},
            {'q': 'How do you handle missing values in data analysis?', 'a': 'Investigate the missingness mechanism (MCAR, MAR, MNAR). Options include deletion (if minimal), imputation (mean/median/mode or KNN/model-based), or creating an indicator flag.'},
            {'q': 'What is A/B Testing and how do you determine statistical significance?', 'a': 'A/B testing compares two versions (A and B) on a random sample. Statistical significance is evaluated via p-value (typically < 0.05) using a Two-Sample t-test or Chi-square test.'},
            {'q': 'How would you calculate Customer Retention Rate / Churn Rate using SQL?', 'a': 'Identify active users in period T0 and track how many make repeat purchases/logins in period T1. Churn = (Lost Customers / Starting Customers) * 100.'}
        ]
    },
    'Software Engineer': {
        'category': 'Software Engineering',
        'degrees': ['B.Tech', 'M.Tech', 'BCA', 'MCA', 'B.Sc'],
        'majors': ['Computer Science', 'Information Technology', 'Electronics'],
        'industries': ['IT', 'Engineering', 'Finance'],
        'core_skills': ['Data Structures', 'Algorithms', 'Java', 'C++', 'Python', 'Git', 'OOP'],
        'advanced_skills': ['System Design', 'DBMS', 'Operating Systems', 'Computer Networks', 'SQL', 'Linux', 'Unit Testing'],
        'roadmap': [
            {'phase': 'Phase 1: DSA and Problem Solving', 'duration': 'Weeks 1-6', 'topics': 'Arrays, Strings, HashMaps, Two Pointers, Trees, Graphs, Dynamic Programming', 'resources': [{'title': 'NeetCode 150 DSA Roadmap', 'url': 'https://neetcode.io/roadmap'}, {'title': 'LeetCode Top Interview 150', 'url': 'https://leetcode.com/studyplan/top-interview-150/'}]},
            {'phase': 'Phase 2: CS Core Fundamentals', 'duration': 'Weeks 7-10', 'topics': 'OOP Principles, Database Design (SQL and NoSQL), OS (Processes, Threads, Concurrency), Networks', 'resources': [{'title': 'CS50 Harvard Computer Science', 'url': 'https://cs50.harvard.edu/x/'}, {'title': 'GeeksforGeeks Core CS', 'url': 'https://www.geeksforgeeks.org'}]},
            {'phase': 'Phase 3: System Design and Production Code', 'duration': 'Weeks 11-16', 'topics': 'Low-Level and High-Level System Design, Microservices, Caching (Redis), Message Queues (Kafka)', 'resources': [{'title': 'System Design Primer (GitHub)', 'url': 'https://github.com/donnemartin/system-design-primer'}, {'title': 'ByteByteGo System Design', 'url': 'https://www.youtube.com/@ByteByteGo'}]}
        ],
        'interview_questions': [
            {'q': 'Explain the difference between Process and Thread in Operating Systems.', 'a': 'A process is an executing instance of a program with its own isolated memory space. A thread is a lightweight execution unit within a process that shares memory and resources with sibling threads.'},
            {'q': 'What are SOLID principles in Object-Oriented Software Engineering?', 'a': 'Single Responsibility, Open/Closed, Liskov Substitution, Interface Segregation, and Dependency Inversion. They ensure scalable, maintainable, and loosely-coupled code.'},
            {'q': 'How does HashMap work internally in Java/C++? What is its time complexity?', 'a': 'HashMap uses an array of buckets and a hash function to map keys to bucket indices. Collisions are handled using Linked Lists or Balanced Trees. Average lookup/insert is O(1), worst case is O(N).'},
            {'q': 'Explain ACID properties in Database Management Systems.', 'a': 'Atomicity (all or nothing), Consistency (preserves integrity rules), Isolation (concurrent transactions do not interfere), and Durability (committed changes persist permanently).'},
            {'q': 'How would you design a rate limiter for an API?', 'a': 'Use Token Bucket or Leaky Bucket algorithm backed by an in-memory store like Redis with atomic increments and key expiry for lightning-fast request throttling.'}
        ]
    },
    'Frontend Developer': {
        'category': 'Web Development',
        'degrees': ['B.Tech', 'BCA', 'MCA', 'B.Sc'],
        'majors': ['Computer Science', 'Information Technology'],
        'industries': ['IT', 'Marketing', 'Education'],
        'core_skills': ['HTML', 'CSS', 'JavaScript', 'React', 'Tailwind'],
        'advanced_skills': ['TypeScript', 'Next.js', 'Vue.js', 'Redux', 'Bootstrap', 'REST API', 'Git', 'Webpack'],
        'roadmap': [
            {'phase': 'Phase 1: Modern Frontend Core', 'duration': 'Weeks 1-4', 'topics': 'HTML5 Semantic Tags, Modern CSS (Flexbox, Grid, Tailwind), JavaScript ES6+ (Promises, Async/Await, DOM)', 'resources': [{'title': 'The Modern JavaScript Tutorial', 'url': 'https://javascript.info'}, {'title': 'MDN Web Docs', 'url': 'https://developer.mozilla.org'}]},
            {'phase': 'Phase 2: React and Ecosystem', 'duration': 'Weeks 5-10', 'topics': 'React Hooks, Component Architecture, State Management (Zustand/Redux), REST API Integration', 'resources': [{'title': 'Official React Documentation', 'url': 'https://react.dev'}, {'title': 'Roadmap.sh Frontend Guide', 'url': 'https://roadmap.sh/frontend'}]},
            {'phase': 'Phase 3: Next.js & Performance', 'duration': 'Weeks 11-16', 'topics': 'Next.js 14 App Router, Server Components, TypeScript, Bundle Optimization, Web Vitals, CI/CD', 'resources': [{'title': 'Next.js Learn Course', 'url': 'https://nextjs.org/learn'}, {'title': 'Web.dev Performance Guides', 'url': 'https://web.dev'}]}
        ],
        'interview_questions': [
            {'q': 'What is the Virtual DOM in React and how does reconciliation work?', 'a': 'Virtual DOM is a lightweight in-memory representation of the real DOM. When state changes, React computes the diff (Diffing Algorithm) and applies only the necessary batch updates to the real DOM.'},
            {'q': 'Explain the JavaScript Event Loop (Call Stack, Microtask Queue, Callback Queue).', 'a': 'The event loop checks if the Call Stack is empty. When empty, it executes Microtasks (Promises/MutationObserver) before executing Callbacks from the Task Queue (setTimeout, I/O).'},
            {'q': 'What is the difference between Server-Side Rendering (SSR) and Client-Side Rendering (CSR)?', 'a': 'In CSR, the browser downloads blank HTML and builds UI via JS. In SSR, the server pre-renders HTML per request, yielding faster First Contentful Paint and superior SEO.'},
            {'q': 'How do you optimize the loading performance of a web application?', 'a': 'Use Code Splitting / Lazy Loading (React.lazy), modern image formats (WebP), CDN caching, tree shaking, and reducing heavy npm dependencies.'},
            {'q': 'What are Closures in JavaScript and where are they used?', 'a': 'A closure is a function bundled together with references to its surrounding lexical environment. Closures enable data encapsulation, currying, and maintaining state in React Hooks.'}
        ]
    },
    'Backend Developer': {
        'category': 'Web & Cloud Engineering',
        'degrees': ['B.Tech', 'MCA', 'BCA', 'M.Tech', 'B.Sc'],
        'majors': ['Computer Science', 'Information Technology'],
        'industries': ['IT', 'Finance', 'Healthcare'],
        'core_skills': ['Node.js', 'Python', 'Java', 'SQL', 'PostgreSQL', 'Express.js', 'Django'],
        'advanced_skills': ['FastAPI', 'Spring Boot', 'MongoDB', 'Redis', 'Microservices', 'Docker', 'REST API', 'Git'],
        'roadmap': [
            {'phase': 'Phase 1: Backend Language & DB', 'duration': 'Weeks 1-4', 'topics': 'Node.js / Python / Java OOP, Relational Databases (PostgreSQL/MySQL), Indexing, Transactions', 'resources': [{'title': 'Node.js Official Documentation', 'url': 'https://nodejs.org/docs'}, {'title': 'PostgreSQL Tutorial', 'url': 'https://www.postgresqltutorial.com'}]},
            {'phase': 'Phase 2: RESTful APIs & Authentication', 'duration': 'Weeks 5-10', 'topics': 'Express/Django/FastAPI, JWT Auth, OAuth2, Middleware, Input Validation, Unit Testing', 'resources': [{'title': 'Roadmap.sh Backend Guide', 'url': 'https://roadmap.sh/backend'}, {'title': 'FastAPI Official Guide', 'url': 'https://fastapi.tiangolo.com'}]},
            {'phase': 'Phase 3: Microservices & Scalability', 'duration': 'Weeks 11-16', 'topics': 'Redis In-Memory Caching, Kafka/RabbitMQ Message Queues, Docker Containerization, API Gateways', 'resources': [{'title': 'Microservices.io Patterns', 'url': 'https://microservices.io'}, {'title': 'Redis University', 'url': 'https://university.redis.com'}]}
        ],
        'interview_questions': [
            {'q': 'How do Database Indexes work, and what are their trade-offs?', 'a': 'Indexes use B-Trees or Hash tables to enable O(log N) lookup without scanning full tables. Trade-off: indexes consume extra storage and slow down INSERT/UPDATE/DELETE write operations.'},
            {'q': 'Explain the difference between SQL and NoSQL databases. When to choose which?', 'a': 'SQL databases (PostgreSQL) are ACID-compliant with rigid relational schemas, ideal for structured transactional data. NoSQL (MongoDB, DynamoDB) offers horizontal scalability and flexible schemas for unstructured data.'},
            {'q': 'How do you handle Distributed Transactions across Microservices?', 'a': 'Use the Saga Pattern (orchestration or choreography with compensating transactions) or Two-Phase Commit (2PC) to preserve eventual consistency without locking databases.'},
            {'q': 'What is JWT Authentication and how is it securely validated?', 'a': 'JWT contains Header, Payload, and Signature. The server verifies the signature using a secret key or RSA public key without querying a database for session state.'},
            {'q': 'How do you protect a backend application against SQL Injection and XSS attacks?', 'a': 'Use Parameterized Queries / ORM prepared statements for SQL injection. For XSS, sanitize and HTML-encode user inputs and configure strict Content-Security-Policy headers.'}
        ]
    },
    'Full Stack Developer': {
        'category': 'Web Development',
        'degrees': ['B.Tech', 'MCA', 'BCA', 'M.Tech'],
        'majors': ['Computer Science', 'Information Technology'],
        'industries': ['IT', 'Consulting', 'Finance'],
        'core_skills': ['React', 'Node.js', 'JavaScript', 'SQL', 'MongoDB', 'HTML', 'CSS'],
        'advanced_skills': ['TypeScript', 'Next.js', 'Express.js', 'PostgreSQL', 'Docker', 'Tailwind', 'Git', 'REST API'],
        'roadmap': [
            {'phase': 'Phase 1: Frontend Mastery', 'duration': 'Weeks 1-5', 'topics': 'Modern HTML5/CSS3, TypeScript, React 18, State Management, Responsive Web Design', 'resources': [{'title': 'Roadmap.sh Full Stack Guide', 'url': 'https://roadmap.sh/full-stack'}, {'title': 'The Odin Project', 'url': 'https://www.theodinproject.com'}]},
            {'phase': 'Phase 2: Backend & Databases', 'duration': 'Weeks 6-10', 'topics': 'Node.js, Express, REST & GraphQL APIs, PostgreSQL & Prisma ORM, MongoDB', 'resources': [{'title': 'Full Stack Open (University of Helsinki)', 'url': 'https://fullstackopen.com'}, {'title': 'Prisma ORM Docs', 'url': 'https://www.prisma.io/docs'}]},
            {'phase': 'Phase 3: Production & DevOps', 'duration': 'Weeks 11-16', 'topics': 'Next.js Full-Stack App, Docker, AWS/Vercel Deployment, CI/CD Pipelines, Testing (Jest/Cypress)', 'resources': [{'title': 'Next.js Production Guide', 'url': 'https://nextjs.org/docs'}, {'title': 'Docker for Developers', 'url': 'https://docs.docker.com'}]}
        ],
        'interview_questions': [
            {'q': 'Explain the end-to-end flow when a user types a URL in a browser and presses Enter.', 'a': 'DNS resolution maps domain to IP -> TCP 3-way handshake -> TLS negotiation -> HTTP GET request -> Load balancer routes to server -> Server renders HTML/API response -> Browser parses HTML/CSS/JS and renders DOM tree.'},
            {'q': 'What are the key architectural differences between Monolith and Microservices?', 'a': 'Monolith bundles all UI, business logic, and DB in a single codebase (easy to start, hard to scale). Microservices decouple domain services with independent deployment and separate DBs (resilient, higher operational complexity).'},
            {'q': 'What is CORS and how do you resolve CORS errors in web development?', 'a': 'Cross-Origin Resource Sharing is a browser security mechanism that blocks requests across different origins unless the backend responds with matching Access-Control-Allow-Origin headers.'},
            {'q': 'How do you structure State Management in a complex Full-Stack application?', 'a': 'Split into Server State (React Query / SWR for caching and syncing API data), Global Client State (Zustand / Redux Toolkit for user session/theme), and Local State (useState/useReducer).'},
            {'q': 'How do you design an optimistic UI update in React?', 'a': 'Immediately update the UI before the network request completes, and revert back to the previous state with a toast notification if the API call returns an error.'}
        ]
    },
    'DevOps Engineer': {
        'category': 'Cloud & Infrastructure',
        'degrees': ['B.Tech', 'M.Tech', 'MCA', 'B.Sc'],
        'majors': ['Computer Science', 'Information Technology', 'Electronics'],
        'industries': ['IT', 'Cloud', 'Finance'],
        'core_skills': ['Linux', 'Docker', 'Kubernetes', 'CI/CD', 'AWS', 'Git', 'Bash'],
        'advanced_skills': ['Terraform', 'Ansible', 'GitHub Actions', 'Jenkins', 'Nginx', 'Prometheus', 'Grafana', 'Python'],
        'roadmap': [
            {'phase': 'Phase 1: Linux, Shell & Networking', 'duration': 'Weeks 1-4', 'topics': 'Linux Admin, Bash Scripting, DNS, HTTP/HTTPS, SSL/TLS, SSH, Firewall Management', 'resources': [{'title': 'Linux Journey Tutorial', 'url': 'https://linuxjourney.com'}, {'title': 'Roadmap.sh DevOps Guide', 'url': 'https://roadmap.sh/devops'}]},
            {'phase': 'Phase 2: Containers & Orchestration', 'duration': 'Weeks 5-10', 'topics': 'Docker Containerization, Multi-stage Builds, Kubernetes Architecture (Pods, Services, Ingress, Deployments)', 'resources': [{'title': 'Docker Official Getting Started', 'url': 'https://docs.docker.com/get-started/'}, {'title': 'Kubernetes Official Documentation', 'url': 'https://kubernetes.io/docs/home/'}]},
            {'phase': 'Phase 3: CI/CD, Cloud & IaC', 'duration': 'Weeks 11-16', 'topics': 'GitHub Actions / Jenkins Pipelines, Terraform Infrastructure as Code, AWS / Azure Cloud Deployment, Monitoring', 'resources': [{'title': 'HashiCorp Terraform Tutorials', 'url': 'https://developer.hashicorp.com/terraform/tutorials'}, {'title': 'TechWorld with Nana (YouTube)', 'url': 'https://www.youtube.com/@TechWorldwithNana'}]}
        ],
        'interview_questions': [
            {'q': 'Explain the difference between a Container and a Virtual Machine.', 'a': 'VMs virtualize hardware and include a full guest OS, making them heavy. Containers virtualize the OS kernel, sharing the host OS, making them lightweight, fast, and portable.'},
            {'q': 'What is Kubernetes Ingress and how does it route traffic?', 'a': 'Ingress is an API object that manages external access to services in a cluster, providing HTTP/HTTPS routing rules, SSL termination, and load balancing.'},
            {'q': 'What is Infrastructure as Code (IaC) and what problems does it solve?', 'a': 'IaC manages infrastructure using declarative code (e.g. Terraform). It eliminates manual configuration drift, enables version control, automated rollback, and reproducible environments.'},
            {'q': 'How does a Blue-Green Deployment strategy work?', 'a': 'Two identical production environments exist (Blue = live, Green = idle). The new release is deployed to Green; once verified, the router/load balancer instantly flips live traffic from Blue to Green with zero downtime.'},
            {'q': 'What is the difference between Continuous Delivery and Continuous Deployment?', 'a': 'In Continuous Delivery, code changes automatically pass automated testing and build, ready for 1-click manual deployment. In Continuous Deployment, every passing change automatically deploys directly to production.'}
        ]
    },
    'Cloud Engineer': {
        'category': 'Cloud & Infrastructure',
        'degrees': ['B.Tech', 'M.Tech', 'MCA', 'B.Sc'],
        'majors': ['Computer Science', 'Information Technology', 'Electronics'],
        'industries': ['IT', 'Cloud', 'Consulting'],
        'core_skills': ['AWS', 'Azure', 'Google Cloud', 'Terraform', 'Linux', 'Networking'],
        'advanced_skills': ['Docker', 'Kubernetes', 'Python', 'CI/CD', 'Security', 'SQL', 'CloudFormation'],
        'roadmap': [
            {'phase': 'Phase 1: Cloud & Networking Foundations', 'duration': 'Weeks 1-4', 'topics': 'VPC, Subnets, Route Tables, NAT Gateways, IAM Roles, Security Groups, DNS & CDN', 'resources': [{'title': 'AWS Skill Builder', 'url': 'https://explore.skillbuilder.aws'}, {'title': 'Microsoft Learn for Azure', 'url': 'https://learn.microsoft.com/en-us/training/azure/'}]},
            {'phase': 'Phase 2: Compute, Storage & Serverless', 'duration': 'Weeks 5-10', 'topics': 'EC2, S3, RDS, DynamoDB, Lambda Serverless, Auto Scaling, Elastic Load Balancer', 'resources': [{'title': 'AWS Certified Solutions Architect Course', 'url': 'https://learn.cantrill.io'}, {'title': 'Cloud Academy', 'url': 'https://cloudacademy.com'}]},
            {'phase': 'Phase 3: Automation, Security & FinOps', 'duration': 'Weeks 11-16', 'topics': 'Terraform Multi-Cloud, CloudWatch, KMS Encryption, Cost Optimization (FinOps), Disaster Recovery', 'resources': [{'title': 'Terraform Associate Certification Guide', 'url': 'https://developer.hashicorp.com/terraform/tutorials/certification'}, {'title': 'AWS Well-Architected Framework', 'url': 'https://aws.amazon.com/architecture/well-architected/'}]}
        ],
        'interview_questions': [
            {'q': 'What is the difference between a Public and Private Subnet in an AWS VPC?', 'a': 'A Public Subnet has a route table entry pointing to an Internet Gateway (IGW), allowing direct inbound/outbound internet traffic. A Private Subnet routes outbound internet traffic only through a NAT Gateway in a public subnet.'},
            {'q': 'Explain AWS Shared Responsibility Model.', 'a': 'AWS is responsible for security OF the cloud (physical data centers, hardware, virtualization layer). The customer is responsible for security IN the cloud (guest OS, IAM permissions, data encryption, firewall rules).'},
            {'q': 'How do you design a High-Availability Multi-Region Disaster Recovery architecture?', 'a': 'Use Multi-AZ deployments for primary data, cross-region asynchronous database replication (e.g. Aurora Global DB, S3 Cross-Region Replication), and Route 53 DNS failover routing.'},
            {'q': 'What are Serverless architectures and what are their trade-offs?', 'a': 'Serverless (e.g. AWS Lambda) scales automatically with zero idle server cost. Trade-offs: cold starts on initialization, execution timeouts (15 mins), and vendor lock-in.'},
            {'q': 'How do you securely manage Secrets and API keys in Cloud Environments?', 'a': 'Use dedicated secret managers like AWS Secrets Manager or HashiCorp Vault with automatic key rotation and IAM role-based temporary credentials instead of hardcoding in code.'}
        ]
    },
    'Project Manager': {
        'category': 'Management & Strategy',
        'degrees': ['MBA', 'BBA', 'B.Tech', 'M.Tech', 'MCA'],
        'majors': ['Business', 'Finance', 'Computer Science', 'Information Technology'],
        'industries': ['Consulting', 'IT', 'Finance', 'Healthcare', 'Marketing'],
        'core_skills': ['Agile', 'Scrum', 'Jira', 'Project Management', 'Team Leadership', 'Stakeholder Management'],
        'advanced_skills': ['Product Management', 'Risk Management', 'Budgeting', 'Communication', 'Excel', 'Business Analysis'],
        'roadmap': [
            {'phase': 'Phase 1: Agile & Scrum Mastery', 'duration': 'Weeks 1-4', 'topics': 'Scrum Ceremonies (Daily Standup, Sprint Planning, Retrospective), Jira Workflows, User Stories & Story Points', 'resources': [{'title': 'Scrum Guide', 'url': 'https://scrumguides.org'}, {'title': 'Atlassian Agile Coach', 'url': 'https://www.atlassian.com/agile'}]},
            {'phase': 'Phase 2: Project Governance & Risk', 'duration': 'Weeks 5-8', 'topics': 'WBS (Work Breakdown Structure), Gantt Charts, Risk Registers, Resource Allocation, Conflict Resolution', 'resources': [{'title': 'PMI PMBOK Guide Basics', 'url': 'https://www.pmi.org'}, {'title': 'Google Project Management Certificate', 'url': 'https://grow.google/certificates/project-management/'}]},
            {'phase': 'Phase 3: Strategy & Executive Leadership', 'duration': 'Weeks 9-14', 'topics': 'Roadmapping, Stakeholder Communication, OKRs and KPIs, Cross-functional Execution, Budget Tracking', 'resources': [{'title': 'ProductPlan Roadmap Guide', 'url': 'https://www.productplan.com'}, {'title': 'Harvard Business Review Leadership Articles', 'url': 'https://hbr.org'}]}
        ],
        'interview_questions': [
            {'q': 'How do you handle a situation where a project is falling behind its scheduled deadline?', 'a': 'Perform a critical path analysis to identify bottlenecks, communicate transparently with stakeholders, evaluate Scope vs Resources (Fast-tracking or Crashing), and renegotiate non-critical deliverables.'},
            {'q': 'What is the difference between Agile and Waterfall methodologies?', 'a': 'Waterfall is a sequential, linear approach with rigid upfront planning. Agile is iterative and flexible, delivering value in 2-4 week sprints with continuous feedback and adaptive changes.'},
            {'q': 'How do you manage disagreements or conflicting priorities between engineering and product stakeholders?', 'a': 'Anchor discussions on data, business impact, and company OKRs. Use frameworks like RICE (Reach, Impact, Confidence, Effort) to objectively score and prioritize feature requests.'},
            {'q': 'What makes an effective Retrospective meeting in Scrum?', 'a': 'Fostering psychological safety, focusing on processes rather than blaming individuals, celebrating wins, and walking away with 2-3 clear, actionable improvements for the next sprint.'},
            {'q': 'How do you define and track project success metrics?', 'a': 'Establish SMART goals upfront. Track Schedule Variance (SV), Cost Variance (CV), Sprint Velocity, Burndown Charts, and post-launch customer adoption / ROI metrics.'}
        ]
    },
    'Business Analyst': {
        'category': 'Management & Strategy',
        'degrees': ['MBA', 'BBA', 'B.Tech', 'B.Com', 'B.Sc'],
        'majors': ['Business', 'Finance', 'Information Technology', 'Computer Science'],
        'industries': ['Consulting', 'Finance', 'IT', 'Marketing', 'Healthcare'],
        'core_skills': ['Business Analysis', 'SQL', 'Excel', 'Power BI', 'Agile', 'Requirements Gathering'],
        'advanced_skills': ['Tableau', 'Jira', 'Financial Modeling', 'Data Analysis', 'Process Mapping', 'UML', 'Wireframing'],
        'roadmap': [
            {'phase': 'Phase 1: Requirements & Business Processes', 'duration': 'Weeks 1-4', 'topics': 'BRD / FRD Documentation, Stakeholder Interviews, Use Cases, User Stories, BPMN Process Modeling', 'resources': [{'title': 'IIBA BABOK Guide Overview', 'url': 'https://www.iiba.org'}, {'title': 'Bridging the Gap BA Roadmap', 'url': 'https://www.bridging-the-gap.com'}]},
            {'phase': 'Phase 2: Data Analysis & BI Tools', 'duration': 'Weeks 5-8', 'topics': 'Advanced Excel Modeling, SQL for Business Queries, Power BI / Tableau Dashboards, KPI Tracking', 'resources': [{'title': 'Microsoft Power BI Guided Learning', 'url': 'https://learn.microsoft.com/en-us/power-bi/'}, {'title': 'SQLZoo Interactive SQL', 'url': 'https://sqlzoo.net'}]},
            {'phase': 'Phase 3: Agile BA & Strategy', 'duration': 'Weeks 9-14', 'topics': 'Product Backlog Refinement, Acceptance Criteria (Gherkin), Gap Analysis, Cost-Benefit Analysis (ROI)', 'resources': [{'title': 'Atlassian Agile BA Guide', 'url': 'https://www.atlassian.com/agile/requirements'}, {'title': 'Coursera Business Analytics Specialization', 'url': 'https://www.coursera.org/specializations/business-analytics'}]}
        ],
        'interview_questions': [
            {'q': 'Explain the difference between Functional and Non-Functional Requirements.', 'a': 'Functional requirements define WHAT the system should do (e.g., user login, process payment). Non-functional requirements define HOW the system behaves (e.g., response time < 200ms, 99.9% uptime, security encryption).'},
            {'q': 'What is a GAP Analysis and how do you conduct it?', 'a': 'GAP Analysis compares the Current State (As-Is) against the Future Desired State (To-Be) to identify gaps, root causes, and necessary action steps/resources needed to bridge them.'},
            {'q': 'How do you handle ambiguous or conflicting requirements from different stakeholders?', 'a': 'Conduct joint alignment workshops, map requirements to business objectives and ROI, clarify edge cases with prototype wireframes, and secure formal stakeholder sign-off.'},
            {'q': 'What is BPMN and why is process mapping important?', 'a': 'Business Process Model and Notation (BPMN) is a standardized visual modeling language. It helps identify operational bottlenecks, redundancies, and automation opportunities.'},
            {'q': 'How do you write effective User Stories with Acceptance Criteria?', 'a': 'Use the format: "As a [user role], I want to [action] so that [business value]". Define acceptance criteria using Given-When-Then (Gherkin syntax) to ensure clear testability.'}
        ]
    },
    'Financial Analyst': {
        'category': 'Finance & Consulting',
        'degrees': ['MBA', 'B.Com', 'BBA', 'M.Sc', 'B.Sc'],
        'majors': ['Finance', 'Business', 'Mathematics', 'Economics'],
        'industries': ['Finance', 'Consulting', 'Banking'],
        'core_skills': ['Financial Modeling', 'Excel', 'Accounting', 'Valuation', 'Corporate Finance'],
        'advanced_skills': ['SQL', 'Power BI', 'Python', 'Risk Management', 'Tableau', 'Statistics', 'DCF Analysis'],
        'roadmap': [
            {'phase': 'Phase 1: Financial Statements & Excel', 'duration': 'Weeks 1-4', 'topics': 'Income Statement, Balance Sheet, Cash Flow Statement 3-Way Linkage, Advanced Excel Shortcuts & Formulas', 'resources': [{'title': 'Corporate Finance Institute (CFI) Free Courses', 'url': 'https://corporatefinanceinstitute.com'}, {'title': 'Khan Academy Finance & Capital Markets', 'url': 'https://www.khanacademy.org/economics-finance-domain/core-finance'}]},
            {'phase': 'Phase 2: Financial Modeling & Valuation', 'duration': 'Weeks 5-9', 'topics': 'Discounted Cash Flow (DCF), Comparable Company Analysis (Comps), Precedent Transactions, LBO Modeling', 'resources': [{'title': 'Wall Street Prep Financial Modeling Guide', 'url': 'https://www.wallstreetprep.com'}, {'title': 'Aswath Damodaran Valuation Lectures (NYU)', 'url': 'https://pages.stern.nyu.edu/~adamodar/'}]},
            {'phase': 'Phase 3: BI & Quantitative Analysis', 'duration': 'Weeks 10-14', 'topics': 'Power BI Financial Dashboards, SQL for Financial Transactions, Python for Portfolio Risk & Variance Analysis', 'resources': [{'title': 'Coursera Financial Engineering', 'url': 'https://www.coursera.org'}, {'title': 'Investopedia Advanced Finance Concepts', 'url': 'https://www.investopedia.com'}]}
        ],
        'interview_questions': [
            {'q': 'Walk me through how the 3 Financial Statements are linked together.', 'a': 'Net income from the Income Statement flows into Retained Earnings on the Balance Sheet and starts the Cash Flow Statement under Operating Cash Flow. Working Capital changes and CapEx on the CFS update Balance Sheet assets/liabilities. Ending Cash on CFS becomes Cash on the Balance Sheet.'},
            {'q': 'How do you calculate and interpret Free Cash Flow (FCF)?', 'a': 'Free Cash Flow to Firm (FCFF) = EBIT*(1-Tax) + D&A - CapEx - Change in Working Capital. It measures the real cash generated available to all capital providers after mandatory reinvestment.'},
            {'q': 'Explain how a Discounted Cash Flow (DCF) model works.', 'a': 'A DCF projects unlevered free cash flows over a 5-10 year horizon, calculates a Terminal Value (via Gordon Growth or Exit Multiple), and discounts all future cash flows to present value using the Weighted Average Cost of Capital (WACC).'},
            {'q': 'If Depreciation increases by $10, how does it affect all three statements (assuming 20% tax rate)?', 'a': 'Income Statement: EBIT drops by $10, Net Income drops by $8 ($10 - $2 tax shield). Cash Flow: Net income drops by $8, add back $10 depreciation -> Cash increases by $2. Balance Sheet: Cash up $2, PP&E down $10 -> Total Assets down $8, Retained Earnings down $8.'},
            {'q': 'What is WACC (Weighted Average Cost of Capital) and how is Cost of Equity calculated?', 'a': 'WACC = (E/V * Ke) + (D/V * Kd * (1-t)). Cost of Equity (Ke) is typically estimated via the CAPM model: Ke = Rf + Beta * (Rm - Rf).' }
        ]
    },
    'Electrical Engineer': {
        'category': 'Core Engineering',
        'degrees': ['B.Tech', 'M.Tech', 'Diploma', 'B.Sc'],
        'majors': ['Electrical', 'Electronics'],
        'industries': ['Engineering', 'Manufacturing', 'Energy'],
        'core_skills': ['Circuit Design', 'PLC', 'MATLAB', 'MATLAB Simulink', 'Power Systems'],
        'advanced_skills': ['Embedded Systems', 'C', 'AutoCAD', 'Control Systems', 'Microcontrollers', 'Python', 'PCB Design'],
        'roadmap': [
            {'phase': 'Phase 1: Circuit Theory & Analysis', 'duration': 'Weeks 1-4', 'topics': 'Ohm’s & Kirchhoff’s Laws, AC/DC Circuit Analysis, Semiconductor Devices, Op-Amps, SPICE Simulation', 'resources': [{'title': 'All About Circuits Textbook', 'url': 'https://www.allaboutcircuits.com/textbook/'}, {'title': 'Khan Academy Electrical Engineering', 'url': 'https://www.khanacademy.org/science/electrical-engineering'}]},
            {'phase': 'Phase 2: Control Systems & Power', 'duration': 'Weeks 5-10', 'topics': 'Feedback Control Systems, MATLAB & Simulink Modeling, Power Grid Fundamentals, Transformers, Electric Motors', 'resources': [{'title': 'Brian Douglas Control Systems (YouTube)', 'url': 'https://www.youtube.com/@BrianBDouglas'}, {'title': 'MathWorks MATLAB Tutorials', 'url': 'https://www.mathworks.com/learn/tutorials/matlab-onramp.html'}]},
            {'phase': 'Phase 3: Automation & Embedded', 'duration': 'Weeks 11-16', 'topics': 'PLC Ladder Logic Programming, SCADA Systems, Embedded C, Microcontrollers (ARM / STM32), PCB Design (KiCAD)', 'resources': [{'title': 'RealPars PLC Automation', 'url': 'https://realpars.com'}, {'title': 'KiCad Official Tutorials', 'url': 'https://www.kicad.org'}]}
        ],
        'interview_questions': [
            {'q': 'What is Power Factor and why is Power Factor Correction important in power systems?', 'a': 'Power factor is the ratio of Real Power (kW) to Apparent Power (kVA). A low power factor draws more reactive current, increasing line losses and equipment strain. Shunt capacitor banks are installed for correction.'},
            {'q': 'Explain the working principle of a 3-Phase Induction Motor.', 'a': 'Supplying balanced 3-phase AC voltage to stator windings produces a rotating magnetic field (RMF). The RMF cuts rotor conductors, inducing EMF and current, creating torque per Lenz\'s Law to rotate at slightly sub-synchronous speed.'},
            {'q': 'How does a PLC (Programmable Logic Controller) scan and execute its program?', 'a': 'A PLC cycle continuously executes 3 steps: 1) Input Scan (reads sensor states), 2) Program Execution (evaluates Ladder Logic sequentially), and 3) Output Scan (updates actuators/relays).'},
            {'q': 'What is the difference between BJT (Bipolar Junction Transistor) and MOSFET?', 'a': 'BJT is a current-controlled device with low input impedance and faster switching in saturation. MOSFET is a voltage-controlled device with extremely high input impedance and lower power dissipation.'},
            {'q': 'Explain the Nyquist Sampling Theorem in signal processing.', 'a': 'To prevent aliasing and accurately reconstruct an analog continuous-time signal, the sampling frequency fs must be at least twice the maximum frequency component present in the signal (fs >= 2 * fmax).' }
        ]
    },
    'Mechanical Engineer': {
        'category': 'Core Engineering',
        'degrees': ['B.Tech', 'M.Tech', 'Diploma', 'B.Sc'],
        'majors': ['Mechanical', 'Automobile'],
        'industries': ['Engineering', 'Manufacturing', 'Automotive'],
        'core_skills': ['AutoCAD', 'SolidWorks', 'Thermodynamics', 'Fluid Mechanics', 'Manufacturing'],
        'advanced_skills': ['ANSYS', 'CATIA', 'MATLAB', 'Robotics', 'C++', 'Python', 'GD&T', 'Finite Element Analysis (FEA)'],
        'roadmap': [
            {'phase': 'Phase 1: Engineering Mechanics & CAD', 'duration': 'Weeks 1-4', 'topics': 'Statics & Dynamics, Strength of Materials, 3D CAD Modeling (SolidWorks / AutoCAD), GD&T Tolerancing', 'resources': [{'title': 'SolidWorks Official Tutorials', 'url': 'https://www.solidworks.com'}, {'title': 'MIT OpenCourseWare Engineering Mechanics', 'url': 'https://ocw.mit.edu'}]},
            {'phase': 'Phase 2: Thermal & Fluid Sciences', 'duration': 'Weeks 5-10', 'topics': '1st & 2nd Laws of Thermodynamics, Heat Transfer (Conduction, Convection, Radiation), Fluid Mechanics, HVAC', 'resources': [{'title': 'Learn Engineering (YouTube)', 'url': 'https://www.youtube.com/@Lesics'}, {'title': 'NPTEL Mechanical Engineering', 'url': 'https://nptel.ac.in'}]},
            {'phase': 'Phase 3: FEA Simulation & Manufacturing', 'duration': 'Weeks 11-16', 'topics': 'Finite Element Analysis (ANSYS Structural/Thermal), CNC Machining, Additive Manufacturing, DFM (Design for Manufacturing)', 'resources': [{'title': 'ANSYS Innovation Courses', 'url': 'https://innovationspace.ansys.com/courses/'}, {'title': 'Autodesk Fusion 360 Tutorials', 'url': 'https://www.autodesk.com'}]}
        ],
        'interview_questions': [
            {'q': 'Explain the Stress-Strain Curve for a ductile material (like mild steel).', 'a': 'The curve shows: 1) Proportional limit (Hooke\'s law), 2) Elastic limit, 3) Yield point (plastic deformation starts), 4) Ultimate Tensile Strength (maximum load), and 5) Fracture/Rupture point with necking.'},
            {'q': 'What is the Second Law of Thermodynamics and how does it relate to Entropy?', 'a': 'The 2nd Law states that heat cannot spontaneously flow from a colder to hotter body without external work. In any spontaneous cyclic process, the total entropy of an isolated system always increases (delta S >= 0).'},
            {'q': 'What is GD&T (Geometric Dimensioning and Tolerancing) and why is it used?', 'a': 'GD&T is a symbolic language specifying the exact allowable variation in geometric characteristics (flatness, concentricity, perpendicularity) to ensure interchangeability and seamless assembly in manufacturing.'},
            {'q': 'Explain the difference between Laminar and Turbulent flow in fluid mechanics.', 'a': 'Laminar flow has smooth, parallel fluid layers (Reynolds Number Re < 2300 in pipes). Turbulent flow exhibits chaotic fluid eddies and high mixing (Re > 4000).'},
            {'q': 'What is the difference between von Mises Stress and Principal Stress in FEA failure criteria?', 'a': 'Principal stresses are the normal stresses on planes where shear stress is zero. Von Mises stress combines all 3D principal stress components into an equivalent scalar stress used to predict yielding in ductile materials.'}
        ]
    },
    'Civil Engineer': {
        'category': 'Core Engineering',
        'degrees': ['B.Tech', 'M.Tech', 'Diploma', 'B.Sc'],
        'majors': ['Civil', 'Structural Engineering'],
        'industries': ['Engineering', 'Construction', 'Infrastructure'],
        'core_skills': ['AutoCAD', 'Structural Analysis', 'Construction Management', 'Surveying', 'STAAD Pro'],
        'advanced_skills': ['Revit', 'Primavera P6', 'Geotechnical Engineering', 'GIS', 'Excel', 'BIM', 'Reinforced Concrete Design'],
        'roadmap': [
            {'phase': 'Phase 1: Surveying & Drafting', 'duration': 'Weeks 1-4', 'topics': 'AutoCAD 2D/3D Drafting, Total Station & GPS Surveying, Leveling, Building Materials & Concrete Technology', 'resources': [{'title': 'Autodesk Civil Design Center', 'url': 'https://www.autodesk.com/solutions/civil-engineering'}, {'title': 'NPTEL Civil Engineering Courses', 'url': 'https://nptel.ac.in'}]},
            {'phase': 'Phase 2: Structural Analysis & Design', 'duration': 'Weeks 5-10', 'topics': 'Bending Moment & Shear Force Diagrams, Reinforced Concrete (IS 456 / ACI), STAAD Pro / ETABS Structural Modeling', 'resources': [{'title': 'Bentley STAAD.Pro Tutorials', 'url': 'https://www.bentley.com'}, {'title': 'The Structural World', 'url': 'https://thestructuralworld.com'}]},
            {'phase': 'Phase 3: BIM & Construction Management', 'duration': 'Weeks 11-16', 'topics': 'Revit Architecture & Structure (BIM), Primavera P6 / MS Project Scheduling, Cost Estimation (BOQ), Geotechnical Soil Mechanics', 'resources': [{'title': 'Coursera Construction Project Management (Columbia Univ)', 'url': 'https://www.coursera.org/learn/construction-project-management'}, {'title': 'Autodesk Revit BIM Learning', 'url': 'https://learn.autodesk.com'}]}
        ],
        'interview_questions': [
            {'q': 'Explain the difference between One-Way Slab and Two-Way Slab in reinforced concrete design.', 'a': 'A One-Way slab is supported on 2 opposite edges or has Ly/Lx >= 2 (bends predominantly in the shorter direction). A Two-Way slab is supported on all 4 sides with Ly/Lx < 2 (bends in both directions).'},
            {'q': 'What is Slump Test of concrete and what does it measure?', 'a': 'The slump test measures the workability and consistency of fresh concrete before placement. Types of slump include True Slump (ideal), Shear Slump (indicates harsh mix), and Collapse Slump (excess water).'},
            {'q': 'What is the importance of Soil Bearing Capacity in foundation engineering?', 'a': 'Bearing capacity is the maximum load per unit area that soil can support without undergoing shear failure or excessive foundation settlement. Footing size is directly determined by Safe Bearing Capacity (SBC).'},
            {'q': 'Explain Critical Path Method (CPM) in construction project scheduling.', 'a': 'CPM identifies the longest sequence of dependent activities with zero total float (slack). Any delay on the critical path directly delays the overall project completion date.'},
            {'q': 'What is Pre-stressed Concrete and what advantages does it have over standard RCC?', 'a': 'Pre-stressed concrete introduces internal compressive stresses using high-strength steel tendons before applying working loads, actively counteracting tensile stresses, allowing longer spans with thinner slabs.'}
        ]
    }
}

DEFAULT_ROLE_INFO: Dict[str, Any] = {
    'category': 'Technology & Innovation',
    'degrees': ['B.Tech', 'M.Tech', 'BCA', 'MCA', 'B.Sc', 'MBA'],
    'majors': ['Computer Science', 'Information Technology', 'Business', 'Finance'],
    'industries': ['IT', 'Data Science', 'Consulting', 'Engineering'],
    'core_skills': ['Problem Solving', 'Data Structures', 'Python', 'SQL', 'Git', 'Communication'],
    'advanced_skills': ['System Architecture', 'Cloud Fundamentals', 'Databases', 'Continuous Learning'],
    'roadmap': [
        {'phase': 'Phase 1: Foundations', 'duration': 'Weeks 1-4', 'topics': 'Core Programming, Version Control, Problem Solving Fundamentals', 'resources': [{'title': 'CS50 Free Online Course', 'url': 'https://cs50.harvard.edu/x/'}]},
        {'phase': 'Phase 2: Domain Specialization', 'duration': 'Weeks 5-10', 'topics': 'Industry Frameworks, Real-World Problem Solving, Database Management', 'resources': [{'title': 'Roadmap.sh Comprehensive Role Guides', 'url': 'https://roadmap.sh'}]},
        {'phase': 'Phase 3: Portfolio & Production', 'duration': 'Weeks 11-16', 'topics': 'End-to-End Deployed Projects, System Architecture, Interview Readiness', 'resources': [{'title': 'FreeCodeCamp Project Tutorials', 'url': 'https://www.freecodecamp.org'}]}
    ],
    'interview_questions': [
        {'q': 'What is your approach to solving a complex technical challenge you haven\'t seen before?', 'a': 'Break the problem down into smaller modular sub-problems, research existing patterns/documentation, build a minimal proof-of-concept, test edge cases, and iteratively refine.'},
        {'q': 'How do you ensure code quality and maintainability in collaborative teams?', 'a': 'Through thorough code reviews, automated unit and integration testing, linting standards, clear documentation, and adhering to Clean Code principles.'},
        {'q': 'Describe a time you optimized a slow system or query.', 'a': 'Profile the system to identify the root bottleneck (I/O, CPU, or Database), implement targeted optimizations like indexing, caching, or algorithmic reduction, and measure before/after throughput.'},
        {'q': 'How do you stay up-to-date with fast-evolving technologies?', 'a': 'Reading engineering blogs (Uber, Netflix, Google), building side projects, following tech roadmaps, and contributing to open-source communities.'},
        {'q': 'Explain how you handle constructive feedback during peer reviews.', 'a': 'View feedback objectively without ego, understand the technical rationale, implement the suggestions, and use it as an opportunity for continuous engineering growth.'}
    ]
}


def get_match_tier(score: float) -> Dict[str, str]:
    """Returns visual match tier badge and colors based on match percentage."""
    if score >= 85.0:
        return {'tier': 'Excellent Match', 'badge': '⭐ Excellent Match', 'color': '#10B981'}
    elif score >= 70.0:
        return {'tier': 'Strong Match', 'badge': '🎯 Strong Match', 'color': '#3B82F6'}
    elif score >= 50.0:
        return {'tier': 'Good Match', 'badge': '👍 Good Match', 'color': '#F59E0B'}
    else:
        return {'tier': 'Moderate Match', 'badge': '📈 Moderate Match', 'color': '#6B7280'}


def calculate_hybrid_score(
    ml_probs: Dict[str, float],
    user_skills_str: str,
    degree: str,
    major: str,
    experience: float,
    industry: str
) -> List[Dict[str, Any]]:
    """
    Computes transparent hybrid match scores across canonical roles.
    Weights:
      - ML Probability: 35%
      - Skill Fit:       40%
      - Academic Fit:    15%
      - Experience Fit:  5%
      - Industry Fit:    5%
    """
    user_skills = [s.strip().lower() for s in (user_skills_str or '').split(',') if s.strip()]
    user_degree = (degree or '').strip().lower()
    user_major = (major or '').strip().lower()
    user_industry = (industry or '').strip().lower()
    exp_val = float(experience or 0)

    results = []

    for role_name in CANONICAL_ROLES:
        role_info = ROLE_TAXONOMY.get(role_name, DEFAULT_ROLE_INFO)
        ml_p = float(ml_probs.get(role_name, 0.0))

        # 1. Skill Fit (40%)
        core_skills = [s.lower() for s in role_info.get('core_skills', [])]
        adv_skills = [s.lower() for s in role_info.get('advanced_skills', [])]

        core_matches = 0
        for cs in core_skills:
            if any(cs in us or us in cs for us in user_skills):
                core_matches += 1

        adv_matches = 0
        for adv in adv_skills:
            if any(adv in us or us in adv for us in user_skills):
                adv_matches += 1

        if user_skills:
            core_ratio = core_matches / max(1, len(core_skills))
            adv_ratio = adv_matches / max(1, len(adv_skills))
            skill_fit = (core_ratio * 0.75) + (adv_ratio * 0.25)
            if core_matches >= 3:
                skill_fit = min(1.0, skill_fit + 0.15)
        else:
            skill_fit = 0.25

        # 2. Academic Fit (15%)
        role_degrees = [d.lower() for d in role_info.get('degrees', [])]
        role_majors = [m.lower() for m in role_info.get('majors', [])]

        deg_match = any(user_degree in rd or rd in user_degree for rd in role_degrees) if user_degree else False
        maj_match = any(user_major in rm or rm in user_major for rm in role_majors) if user_major else False

        if deg_match and maj_match:
            academic_fit = 1.0
        elif deg_match or maj_match:
            academic_fit = 0.75
        elif not user_degree and not user_major:
            academic_fit = 0.5
        else:
            academic_fit = 0.2

        # 3. Experience Fit (5%)
        if exp_val >= 0:
            exp_fit = min(1.0, 0.7 + (min(exp_val, 5.0) * 0.06))
        else:
            exp_fit = 0.5

        # 4. Industry Fit (5%)
        role_industries = [ind.lower() for ind in role_info.get('industries', [])]
        if user_industry and any(user_industry in ri or ri in user_industry for ri in role_industries):
            industry_fit = 1.0
        elif not user_industry:
            industry_fit = 0.6
        else:
            industry_fit = 0.35

        # Calculate weighted composite score
        composite_score = (
            (HYBRID_WEIGHTS['ml_probability'] * ml_p) +
            (HYBRID_WEIGHTS['skill_fit'] * skill_fit) +
            (HYBRID_WEIGHTS['academic_fit'] * academic_fit) +
            (HYBRID_WEIGHTS['experience_fit'] * exp_fit) +
            (HYBRID_WEIGHTS['industry_fit'] * industry_fit)
        )

        match_percentage = composite_score * 100.0

        if ml_p > 0.40 and skill_fit > 0.40:
            match_percentage = min(98.5, max(match_percentage, 85.0 + (ml_p * 10.0)))
        elif ml_p > 0.70:
            match_percentage = min(98.8, max(match_percentage, 88.0 + (ml_p * 10.0)))

        match_percentage = round(float(np.clip(match_percentage, 5.0, 98.8)), 1)
        tier_info = get_match_tier(match_percentage)

        results.append({
            'role': role_name,
            'category': role_info.get('category', 'Technology'),
            'career_match_score': match_percentage,
            'confidence': match_percentage / 100.0,
            'ml_probability': round(ml_p * 100.0, 1),
            'skill_fit_score': round(skill_fit * 100.0, 1),
            'academic_fit_score': round(academic_fit * 100.0, 1),
            'tier': tier_info['tier'],
            'badge': tier_info['badge'],
            'color': tier_info['color']
        })

    results.sort(key=lambda x: x['career_match_score'], reverse=True)
    return results


def analyze_skill_gap(predicted_role: str, user_skills_str: str) -> Dict[str, Any]:
    """Computes matched vs missing skills, readiness score, and includes curated roadmap + interview questions."""
    user_skills = [s.strip().lower() for s in (user_skills_str or '').split(',') if s.strip()]
    role_key = None

    for key in ROLE_TAXONOMY:
        if key.lower() == (predicted_role or '').strip().lower():
            role_key = key
            break

    if not role_key:
        for key in ROLE_TAXONOMY:
            if key.lower() in (predicted_role or '').lower() or (predicted_role or '').lower() in key.lower():
                role_key = key
                break

    role_info = ROLE_TAXONOMY.get(role_key, DEFAULT_ROLE_INFO)
    all_required = role_info['core_skills'] + role_info['advanced_skills']
    matched = []
    missing = []

    for req in all_required:
        req_clean = req.lower()
        if any(req_clean in us or us in req_clean for us in user_skills):
            matched.append(req)
        else:
            missing.append(req)

    total = len(all_required)
    match_count = len(matched)
    readiness = int(round((match_count / max(1, total)) * 100))
    readiness = max(25, min(95, readiness if user_skills else 40))
    boost_val = min(45, max(15, int(len(missing[:3]) * 12)))

    return {
        'predicted_role': predicted_role,
        'category': role_info.get('category', 'Technology'),
        'matched_skills': matched[:8] if matched else ['Analytical Thinking (Baseline)'],
        'missing_skills': missing[:5],
        'readiness_percentage': readiness,
        'potential_boost': f'+{boost_val}% Match Potential',
        'roadmap': role_info.get('roadmap', DEFAULT_ROLE_INFO['roadmap']),
        'interview_questions': role_info.get('interview_questions', DEFAULT_ROLE_INFO['interview_questions'])
    }
