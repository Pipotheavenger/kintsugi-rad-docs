// Renders the code-map diagrams (.kmap) at natural size, with clickable boxes.
import mermaid from "https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs";

const dark = document.body.getAttribute("data-md-color-scheme") === "slate";
mermaid.initialize({
  startOnLoad: false,
  securityLevel: "loose", // needed for `click` links to the code
  theme: dark ? "dark" : "neutral",
  themeVariables: { fontSize: "15px" },
  flowchart: { useMaxWidth: false, htmlLabels: true, wrappingWidth: 340, nodeSpacing: 28, rankSpacing: 48 },
});
await mermaid.run({ querySelector: ".kmap" });
