#!/usr/bin/env python3
"""
TSL TUI Styles Module
====================
Contains all CSS styling for the Textual TUI application.

Architecture:
- TSL_CSS: Main application styles
- COLOR_SCHEME: Color constants for consistent theming

Usage:
    from tools.styles import TSL_CSS, COLOR_SCHEME
"""

# Color scheme constants
COLOR_SCHEME = {
    # Backgrounds
    "bg_primary": "#0b1220",
    "bg_secondary": "#0f172a",
    "bg_tertiary": "#111827",
    "bg_hero": "#0b1b36",
    "bg_config": "#0f2436",
    "bg_preview": "#0d1b2f",
    "bg_output": "#0a1628",
    
    # Accents
    "accent_blue": "#1e3a5f",
    "accent_bright": "#2b5f9e",
    "accent_header": "#11203a",
    
    # Text
    "text_primary": "#e0ecff",
    "text_secondary": "#dbeafe",
    "text_dim": "#bfdbfe",
    "text_highlight": "#93c5fd",
    
    # Status
    "status_success": "#22c55e",
    "status_warning": "#f59e0b",
    "status_error": "#ef4444",
    "status_info": "#3b82f6",
}


# Main CSS styles for TSL TUI (hardcoded for reliability)
TSL_CSS = """
/* ========================================================================
   BASE STYLES
   ======================================================================== */

Screen {
    background: #0b1220;
}

Header {
    background: #11203a;
    color: #dbeafe;
}

Footer {
    background: #11203a;
    color: #dbeafe;
}

/* ========================================================================
   LEFT PANEL - Navigation & Configuration
   ======================================================================== */

#left-panel {
    width: 44;
    height: 100%;
    border: solid #1e3a5f;
    background: #0f172a;
    padding: 0 1;
}

/* Hero Section */
#hero {
    margin: 1 0;
    padding: 1;
    border: round #1d4ed8;
    background: #0b1b36;
    color: #e0ecff;
}

#hero-title {
    text-style: bold;
    color: #93c5fd;
}

#hero-sub {
    color: #bfdbfe;
}

/* Section Headers */
.section-header {
    text-style: bold;
    margin: 1 0 0 0;
    padding: 0 1;
    color: #93c5fd;
}

/* Presets Container */
#presets-container {
    height: auto;
    padding: 1;
    background: #132138;
    border: round #2b5f9e;
    margin: 0 1 1 1;
}

.preset-btn {
    width: 100%;
    margin: 0 0 1 0;
}

.preset-btn.active-preset {
    border: tall #93c5fd;
    text-style: bold;
}

/* Config Inputs */
#config-container {
    height: auto;
    padding: 1;
    background: #0f2436;
    border: round #2b5f9e;
    margin: 0 1 1 1;
}

Input.config-input {
    width: 100%;
    margin: 0 0 1 0;
}

Select {
    margin: 0 0 1 0;
}

/* Quick Actions */
#quick-actions {
    height: auto;
    padding: 1;
    layout: horizontal;
    margin: 0 1 1 1;
    border: round #2b5f9e;
    background: #132138;
}

#other-actions {
    height: auto;
    padding: 1;
    margin: 0 1 1 1;
    border: round #2b5f9e;
    background: #132138;
}

.minor-action {
    margin: 0 0 1 0;
}

/* ========================================================================
   RIGHT PANEL - Output & Preview
   ======================================================================== */

#right-panel {
    width: 1fr;
    height: 100%;
    border: solid #1e3a5f;
    background: #111827;
    padding: 0 1;
}

#output-container {
    height: 1fr;
    padding: 0 0 1 0;
}

#preview-box {
    margin: 1 0;
    height: auto;
    border: round #2b5f9e;
    background: #0d1b2f;
    padding: 1;
}

#preview-label {
    color: #93c5fd;
    text-style: bold;
}

#preview-cmd {
    color: #bfdbfe;
    padding: 0 1;
}

Log#output-log {
    height: 100%;
    border: round #2b5f9e;
    background: #0a1628;
}

/* ========================================================================
   STATUS BAR
   ======================================================================== */

#status-bar {
    height: 3;
    padding: 0 1;
    background: #1e3a5f;
    color: #e2e8f0;
    border: round #60a5fa;
}

/* ========================================================================
   BUTTONS
   ======================================================================== */

Button {
    width: 100%;
}

.action-btn {
    width: 50%;
}

.running-indicator {
    color: #f59e0b;
    text-style: bold;
}
"""


# Export for easy access
__all__ = ['TSL_CSS', 'COLOR_SCHEME']
