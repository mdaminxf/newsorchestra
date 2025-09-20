import React from "react";
import { createRoot } from "react-dom/client";
import App from "./App.jsx"; 

const container = document.getElementById("root");
if (!container) throw new Error("Root container not found");

if (!window._reactRoot) {
  window._reactRoot = createRoot(container);
}

window._reactRoot.render(<App />);
