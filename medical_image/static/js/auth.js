// ─── auth.js — Session-based auth guard ───────────────────────────────────

const AUTH_KEY = 'xhvit_user';

// Pages that require login
const PROTECTED_PAGES = [
  'diagnose.html',
  'dashboard.html',
  'reports.html',
  'report_detail.html',
  'history.html',
  'metrics.html',
  'research.html',
  'services.html',
];

// ── Helpers ────────────────────────────────────────────────────────────────
function getUser() {
  try { return JSON.parse(sessionStorage.getItem(AUTH_KEY)); } catch { return null; }
}
function setUser(user) {
  sessionStorage.setItem(AUTH_KEY, JSON.stringify(user));
}
function clearUser() {
  sessionStorage.removeItem(AUTH_KEY);
}
function isLoggedIn() {
  return !!getUser();
}
function currentPage() {
  return window.location.pathname.split('/').pop() || 'index.html';
}

// ── Guard: redirect to login if not authenticated ─────────────────────────
function guardPage() {
  const page = currentPage();
  if (PROTECTED_PAGES.includes(page) && !isLoggedIn()) {
    // Save intended destination
    sessionStorage.setItem('xhvit_redirect', page);
    window.location.href = 'login.html?next=' + encodeURIComponent(page);
  }
}

// ── Login ─────────────────────────────────────────────────────────────────
function loginUser(name, email, role) {
  setUser({ name, email, role, loginTime: Date.now() });
  const next = new URLSearchParams(window.location.search).get('next')
    || sessionStorage.getItem('xhvit_redirect')
    || 'dashboard.html';
  sessionStorage.removeItem('xhvit_redirect');
  window.location.href = next;
}

// ── Logout ────────────────────────────────────────────────────────────────
function logoutUser() {
  clearUser();
  window.location.href = 'index.html';
}

// ── Update nav dynamically based on auth state ────────────────────────────
function updateNav() {
  const user = getUser();
  const ctaEl = document.querySelector('.nav-cta');
  if (!ctaEl) return;

  if (user) {
    // Logged-in state: show avatar + logout
    const initials = user.name.split(' ').map(n => n[0]).join('').toUpperCase().slice(0,2);
    ctaEl.innerHTML = `
      <a href="dashboard.html" class="btn btn-ghost" style="display:flex;align-items:center;gap:8px">
        <div style="width:26px;height:26px;border-radius:50%;background:linear-gradient(135deg,var(--teal),var(--blue));display:flex;align-items:center;justify-content:center;font-family:var(--font-mono);font-size:11px;font-weight:700;color:var(--bg)">${initials}</div>
        ${user.name.split(' ')[0]}
      </a>
      <button class="btn btn-outline" onclick="logoutUser()" style="font-size:13px">Sign Out</button>
    `;
  } else {
    // Guest state: show login + register
    ctaEl.innerHTML = `
      <a href="login.html" class="btn btn-ghost">Login</a>
      <a href="register.html" class="btn btn-primary">Get Started</a>
    `;
  }
}

// ── Lock protected nav links for guests ───────────────────────────────────
function lockNavLinks() {
  if (isLoggedIn()) return;
  document.querySelectorAll('.nav-links a').forEach(a => {
    const href = a.getAttribute('href');
    if (PROTECTED_PAGES.includes(href)) {
      a.addEventListener('click', e => {
        e.preventDefault();
        showAuthToast(href);
      });
      a.style.opacity = '0.5';
      a.title = 'Login required';
    }
  });
}

// ── Lock protected buttons/links anywhere on public pages ─────────────────
function lockProtectedLinks() {
  if (isLoggedIn()) return;
  document.querySelectorAll('a[href]').forEach(a => {
    const href = a.getAttribute('href');
    if (PROTECTED_PAGES.includes(href)) {
      a.addEventListener('click', e => {
        e.preventDefault();
        showAuthToast(href);
      });
    }
  });
}

// ── Toast notification ────────────────────────────────────────────────────
function showAuthToast(dest) {
  // Remove existing
  document.getElementById('authToast')?.remove();

  const toast = document.createElement('div');
  toast.id = 'authToast';
  toast.innerHTML = `
    <div style="display:flex;align-items:center;gap:14px">
      <div style="font-size:22px">🔒</div>
      <div>
        <div style="font-family:var(--font-head);font-size:15px;font-weight:700;margin-bottom:3px">Login Required</div>
        <div style="font-size:13px;color:rgba(255,255,255,0.7)">Please sign in to access this feature.</div>
      </div>
    </div>
    <div style="display:flex;gap:10px;margin-top:14px">
      <a href="login.html?next=${encodeURIComponent(dest)}" style="flex:1;text-align:center;padding:9px;border-radius:8px;background:var(--teal);color:var(--bg);font-family:var(--font-head);font-size:13px;font-weight:700;cursor:pointer">Login</a>
      <a href="register.html" style="flex:1;text-align:center;padding:9px;border-radius:8px;background:transparent;border:1px solid rgba(255,255,255,0.2);color:#fff;font-size:13px;cursor:pointer">Register</a>
    </div>
    <button onclick="document.getElementById('authToast').remove()" style="position:absolute;top:12px;right:14px;background:none;border:none;color:rgba(255,255,255,0.4);font-size:18px;cursor:pointer;line-height:1">×</button>
  `;
  Object.assign(toast.style, {
    position:'fixed', bottom:'28px', right:'28px', zIndex:'9999',
    background:'#0f1e38', border:'1px solid rgba(0,229,176,0.25)',
    borderRadius:'14px', padding:'20px 22px', maxWidth:'320px', width:'calc(100% - 56px)',
    boxShadow:'0 20px 60px rgba(0,0,0,0.5)', animation:'toastIn 0.3s ease',
    position:'fixed'
  });

  // Add keyframe if not present
  if (!document.getElementById('toastStyle')) {
    const s = document.createElement('style');
    s.id = 'toastStyle';
    s.textContent = '@keyframes toastIn{from{opacity:0;transform:translateY(20px)}to{opacity:1;transform:translateY(0)}}';
    document.head.appendChild(s);
  }

  document.body.appendChild(toast);
  setTimeout(() => toast?.remove(), 6000);
}

// ── Run on every page load ────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  guardPage();
  updateNav();
  lockNavLinks();
  lockProtectedLinks();
});
