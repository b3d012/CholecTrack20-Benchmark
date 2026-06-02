const navToggle = document.querySelector(".nav-toggle");
const navLinks = document.querySelector(".nav-links");
const year = document.querySelector("#year");
const typingText = document.querySelector("#typing-text");
const revealEls = document.querySelectorAll(".reveal");
const sectionLinks = document.querySelectorAll(".nav-links a");

if (year) {
  year.textContent = new Date().getFullYear();
}

if (navToggle && navLinks) {
  navToggle.addEventListener("click", () => {
    const isOpen = document.body.classList.toggle("is-open");
    navToggle.setAttribute("aria-expanded", String(isOpen));
  });

  navLinks.addEventListener("click", (event) => {
    if (event.target.closest("a")) {
      document.body.classList.remove("is-open");
      navToggle.setAttribute("aria-expanded", "false");
    }
  });
}

const revealObserver = new IntersectionObserver(
  (entries) => {
    entries.forEach((entry) => {
      if (entry.isIntersecting) {
        entry.target.classList.add("visible");
        revealObserver.unobserve(entry.target);
      }
    });
  },
  { threshold: 0.12 }
);

revealEls.forEach((el) => revealObserver.observe(el));

const sectionObserver = new IntersectionObserver(
  (entries) => {
    entries.forEach((entry) => {
      if (!entry.isIntersecting) return;
      sectionLinks.forEach((link) => {
        const targetId = link.getAttribute("href");
        link.classList.toggle("active", targetId === `#${entry.target.id}`);
      });
    });
  },
  { threshold: 0.45, rootMargin: "-10% 0px -45% 0px" }
);

document.querySelectorAll("main section[id]").forEach((section) => sectionObserver.observe(section));

const typingPhrases = [
  "benchmark ready",
  "portfolio friendly",
  "compute-aware results",
  "reproducible workflow",
];

if (typingText) {
  let phraseIndex = 0;
  let charIndex = 0;
  let deleting = false;

  const tick = () => {
    const phrase = typingPhrases[phraseIndex];
    typingText.textContent = phrase.slice(0, charIndex);

    if (!deleting && charIndex < phrase.length) {
      charIndex += 1;
    } else if (deleting && charIndex > 0) {
      charIndex -= 1;
    } else {
      deleting = !deleting;
      if (!deleting) {
        phraseIndex = (phraseIndex + 1) % typingPhrases.length;
      }
    }

    const delay = deleting ? 42 : 66;
    setTimeout(tick, deleting ? delay : charIndex === phrase.length ? 900 : delay);
  };

  tick();
}
