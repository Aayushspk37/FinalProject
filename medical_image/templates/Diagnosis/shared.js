// ─── shared.js ───────────────────────────────────────
// Scroll reveal
const revealObserver = new IntersectionObserver(entries => {
  entries.forEach(e => { if (e.isIntersecting) e.target.classList.add('visible'); });
}, { threshold: 0.08 });
document.querySelectorAll('.reveal').forEach(el => revealObserver.observe(el));

// Nav scroll shadow
window.addEventListener('scroll', () => {
  document.querySelector('nav').style.boxShadow = window.scrollY > 40 ? '0 4px 40px rgba(0,0,0,0.5)' : 'none';
});

// Mobile nav
const hamburger = document.getElementById('hamburger');
if (hamburger) {
  let menu = document.getElementById('mobileMenu');
  if (!menu) {
    menu = document.createElement('div');
    menu.className = 'mobile-menu'; menu.id = 'mobileMenu';
    const links = document.querySelector('.nav-links');
    if (links) links.querySelectorAll('a').forEach(a => { const c = a.cloneNode(true); menu.appendChild(c); });
    document.body.appendChild(menu);
  }
  hamburger.addEventListener('click', () => {
    menu.classList.toggle('open');
    const s = hamburger.querySelectorAll('span');
    if (menu.classList.contains('open')) {
      s[0].style.transform='rotate(45deg) translate(5px,5px)'; s[1].style.opacity='0'; s[2].style.transform='rotate(-45deg) translate(5px,-5px)';
    } else { s[0].style.transform=s[2].style.transform=''; s[1].style.opacity=''; }
  });
}
