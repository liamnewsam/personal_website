function goToProject(url) {
  window.location.href = url;
}

// Optional: fade-in animations when scrolling
document.addEventListener("scroll", () => {
  const sections = document.querySelectorAll(".section, .project-card");
  sections.forEach((sec) => {
    const rect = sec.getBoundingClientRect();
    if (rect.top < window.innerHeight - 50) {
      sec.classList.add("visible");
    }
  });
});
