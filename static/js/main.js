// ===== Navbar Scroll Effect =====
const navbar = document.getElementById('navbar');
window.addEventListener('scroll', () => {
  navbar.classList.toggle('scrolled', window.scrollY > 50);
});

// ===== Mobile Menu =====
const hamburgerBtn = document.getElementById('hamburgerBtn');
const mobileMenu = document.getElementById('mobileMenu');
const mobileClose = document.getElementById('mobileClose');

hamburgerBtn.addEventListener('click', () => mobileMenu.classList.add('active'));
mobileClose.addEventListener('click', () => mobileMenu.classList.remove('active'));
function closeMobile() { mobileMenu.classList.remove('active'); }

// ===== Scroll Animations (IntersectionObserver) =====
const fadeEls = document.querySelectorAll('.fade-in');
const observer = new IntersectionObserver((entries) => {
  entries.forEach(entry => {
    if (entry.isIntersecting) {
      entry.target.classList.add('visible');
    }
  });
}, { threshold: 0.1, rootMargin: '0px 0px -50px 0px' });
fadeEls.forEach(el => observer.observe(el));

// ===== Particles =====
function createParticles() {
  const container = document.getElementById('particles');
  if (!container) return;
  for (let i = 0; i < 30; i++) {
    const p = document.createElement('div');
    p.className = 'particle';
    p.style.left = Math.random() * 100 + '%';
    p.style.animationDuration = (8 + Math.random() * 15) + 's';
    p.style.animationDelay = Math.random() * 10 + 's';
    p.style.width = p.style.height = (2 + Math.random() * 3) + 'px';
    const colors = ['var(--accent-blue)', 'var(--accent-purple)', 'var(--accent-pink)', 'var(--accent-cyan)'];
    p.style.background = colors[Math.floor(Math.random() * colors.length)];
    container.appendChild(p);
  }
}
createParticles();

// ===== Smooth Scroll for anchor links =====
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
  anchor.addEventListener('click', function (e) {
    const target = document.querySelector(this.getAttribute('href'));
    if (target) {
      e.preventDefault();
      target.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
  });
});

// ===== Dashboard Bar Animation on Scroll =====
const heroVisual = document.querySelector('.hero-dashboard');
if (heroVisual) {
  const barObserver = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        const fills = entry.target.querySelectorAll('.bar-fill');
        fills.forEach((fill, i) => {
          fill.style.width = '0%';
          setTimeout(() => {
            const widths = ['85%', '72%', '60%', '90%', '78%'];
            fill.style.width = widths[i] || '70%';
          }, 200 + i * 150);
        });
      }
    });
  }, { threshold: 0.3 });
  barObserver.observe(heroVisual);
}

// ===== Live Prediction Demo =====
const predictBtn = document.getElementById('predictBtn');
const demoResult = document.getElementById('demoResult');

if (predictBtn) {
  predictBtn.addEventListener('click', () => {
    const degree = document.getElementById('demo-degree').value;
    const major = document.getElementById('demo-major').value;
    const cgpa = document.getElementById('demo-cgpa').value;
    const exp = document.getElementById('demo-exp').value;
    const skills = document.getElementById('demo-skills').value;
    const industry = document.getElementById('demo-industry').value;

    if (!degree || !major || !skills) {
      demoResult.innerHTML = `
        <div class="result-placeholder">
          <div class="icon">⚠️</div>
          <p style="color:#f59e0b">Please fill in at least Degree, Major, and Skills to get a prediction.</p>
        </div>`;
      return;
    }

    // Show loading
    demoResult.innerHTML = `
      <div class="result-placeholder">
        <div class="icon" style="animation:pulse 1s infinite">🔄</div>
        <p>Analyzing your profile with AI...</p>
      </div>`;

    // Simulate ML prediction (client-side demo)
    setTimeout(() => {
      const result = simulatePrediction(degree, major, skills, cgpa, exp, industry);
      showResult(result);
    }, 1500);
  });
}

function simulatePrediction(degree, major, skills, cgpa, exp, industry) {
  const skillsLower = skills.toLowerCase();
  const roleScores = {};

  // Skill-based scoring
  const skillMap = {
    'ML Engineer': ['machine learning', 'deep learning', 'tensorflow', 'pytorch', 'neural', 'nlp', 'computer vision', 'ai'],
    'Data Scientist': ['data science', 'statistics', 'pandas', 'numpy', 'r programming', 'visualization', 'data analysis', 'jupyter'],
    'Software Developer': ['java', 'c++', 'software', 'algorithms', 'oop', 'git', 'agile', 'design patterns'],
    'Web Developer': ['html', 'css', 'javascript', 'react', 'node', 'angular', 'vue', 'frontend', 'backend', 'web'],
    'Data Analyst': ['excel', 'sql', 'tableau', 'power bi', 'analytics', 'reporting', 'data analysis'],
    'DevOps Engineer': ['docker', 'kubernetes', 'ci/cd', 'aws', 'azure', 'linux', 'terraform', 'devops', 'cloud'],
    'Cloud Architect': ['aws', 'azure', 'gcp', 'cloud', 'microservices', 'serverless'],
    'Cybersecurity Analyst': ['security', 'ethical hacking', 'penetration', 'firewall', 'encryption', 'cybersecurity'],
    'Database Administrator': ['sql', 'mysql', 'postgresql', 'mongodb', 'database', 'oracle', 'redis'],
    'AI Research Scientist': ['research', 'deep learning', 'ai', 'reinforcement learning', 'transformers', 'gpt']
  };

  Object.entries(skillMap).forEach(([role, keywords]) => {
    let score = 0;
    keywords.forEach(kw => { if (skillsLower.includes(kw)) score += 12; });
    if (score > 0) roleScores[role] = score;
  });

  // Major bonus
  const majorBonus = {
    'Computer Science': { 'Software Developer': 10, 'ML Engineer': 8, 'Web Developer': 7 },
    'Data Science': { 'Data Scientist': 15, 'ML Engineer': 10, 'Data Analyst': 8 },
    'Artificial Intelligence': { 'ML Engineer': 15, 'AI Research Scientist': 12, 'Data Scientist': 8 },
    'Information Technology': { 'Web Developer': 10, 'DevOps Engineer': 8, 'Software Developer': 7 },
    'Electronics': { 'DevOps Engineer': 5, 'Software Developer': 4 },
    'Mathematics': { 'Data Scientist': 10, 'ML Engineer': 8, 'Data Analyst': 7 },
    'Business': { 'Data Analyst': 10, 'Database Administrator': 5 }
  };
  if (majorBonus[major]) {
    Object.entries(majorBonus[major]).forEach(([r, s]) => {
      roleScores[r] = (roleScores[r] || 0) + s;
    });
  }

  // CGPA and experience bonus
  const cgpaVal = parseFloat(cgpa) || 7;
  const expVal = parseInt(exp) || 0;
  Object.keys(roleScores).forEach(r => {
    roleScores[r] += cgpaVal * 1.5 + expVal * 2;
  });

  // If no matches, add defaults
  if (Object.keys(roleScores).length === 0) {
    roleScores['Software Developer'] = 45 + cgpaVal * 2;
    roleScores['Data Analyst'] = 35 + cgpaVal * 1.5;
    roleScores['Web Developer'] = 30 + cgpaVal;
  }

  // Normalize to confidence
  const sorted = Object.entries(roleScores).sort((a, b) => b[1] - a[1]).slice(0, 5);
  const maxScore = sorted[0][1];
  const results = sorted.map(([role, score]) => ({
    role,
    confidence: Math.min(98, Math.max(30, (score / maxScore) * 95 + Math.random() * 5)).toFixed(1)
  }));

  return { topRole: results[0].role, topConfidence: results[0].confidence, all: results };
}

function showResult(result) {
  const colors = ['blue', 'purple', 'pink', 'cyan', 'green'];
  const barsHTML = result.all.map((r, i) => `
    <div class="result-bar-item">
      <div class="bar-info"><span>${r.role}</span><span>${r.confidence}%</span></div>
      <div class="bar-track"><div class="bar-fill ${colors[i % colors.length]}" style="width:0%;transition:width 1s ease ${i * 0.2}s"></div></div>
    </div>
  `).join('');

  demoResult.innerHTML = `
    <div class="result-card">
      <div style="font-size:2.5rem;margin-bottom:0.5rem">🎯</div>
      <div class="predicted-role">${result.topRole}</div>
      <div class="confidence-score">Confidence: ${result.topConfidence}%</div>
      <div class="result-bars">${barsHTML}</div>
      <a href="/dashboard" class="btn btn-primary" style="width:100%;justify-content:center;margin-top:1.5rem">Get Full Analysis →</a>
    </div>`;

  // Animate bars
  requestAnimationFrame(() => {
    setTimeout(() => {
      result.all.forEach((r, i) => {
        const bar = demoResult.querySelectorAll('.bar-fill')[i];
        if (bar) bar.style.width = r.confidence + '%';
      });
    }, 50);
  });
}

// ===== Counter Animation =====
function animateCounters() {
  const stats = document.querySelectorAll('.hero-stat h3');
  stats.forEach(stat => {
    const text = stat.textContent;
    const match = text.match(/(\d+)/);
    if (!match) return;
    const target = parseInt(match[0]);
    const suffix = text.replace(match[0], '');
    let current = 0;
    const step = Math.ceil(target / 60);
    const timer = setInterval(() => {
      current += step;
      if (current >= target) { current = target; clearInterval(timer); }
      stat.textContent = current + suffix;
    }, 25);
  });
}

// Run counters when hero is visible
const heroSection = document.getElementById('hero');
if (heroSection) {
  const counterObserver = new IntersectionObserver((entries) => {
    if (entries[0].isIntersecting) {
      animateCounters();
      counterObserver.disconnect();
    }
  }, { threshold: 0.3 });
  counterObserver.observe(heroSection);
}
