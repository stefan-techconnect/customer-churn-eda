document.addEventListener("pointermove", (event) => {
    document.querySelectorAll(".spotlight").forEach((card) => {
        const rect = card.getBoundingClientRect();
        card.style.setProperty("--mx", `${event.clientX - rect.left}px`);
        card.style.setProperty("--my", `${event.clientY - rect.top}px`);
    });
});
