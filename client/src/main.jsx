import React from "react";
import { createRoot } from "react-dom/client";
import { BrowserRouter as Router } from "react-router-dom";
import App from "./App";
import Auth0ProviderWithHistory from "./auth/Auth0Provider";

const container = document.getElementById("root")
const root = createRoot(container)
root.render(
    <Router>
        <Auth0ProviderWithHistory>
            <App />
        </Auth0ProviderWithHistory>
  </Router>,
)