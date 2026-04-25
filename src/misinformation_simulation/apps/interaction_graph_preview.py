from __future__ import annotations

from html import escape

import streamlit as st


def render_graph_preview(node_forms: list[dict[str, str]]) -> None:
    if not node_forms:
        st.info("Add nodes to preview the interaction chain.")
        return

    preview_parts = [
        """
        <style>
        body {
            margin: 0;
            background: transparent;
        }
        .graph-preview-shell {
            --preview-bg: var(--background-color, transparent);
            --preview-node-bg: var(--secondary-background-color, rgba(128, 128, 128, 0.08));
            --preview-border: color-mix(in srgb, var(--text-color, currentColor) 22%, transparent);
            --preview-text: var(--text-color, currentColor);
            --preview-muted: color-mix(in srgb, var(--text-color, currentColor) 72%, transparent);
            --preview-accent: var(--primary-color, #ff4b4b);
            --preview-accent-soft: color-mix(
                in srgb,
                var(--primary-color, #ff4b4b) 16%,
                var(--background-color, transparent)
            );
            border: 1px solid var(--preview-border);
            border-radius: 8px;
            padding: 1rem;
            background: var(--preview-bg);
            color: var(--preview-text);
        }
        .graph-preview-chain {
            display: grid;
            gap: 1rem;
        }
        .graph-preview-row {
            display: flex;
            align-items: stretch;
            width: 100%;
        }
        .graph-preview-unit {
            display: flex;
            flex: 1 1 0;
            min-width: 0;
        }
        .graph-preview-node {
            width: 100%;
            border: 1px solid var(--preview-border);
            border-radius: 8px;
            background: var(--preview-node-bg);
            min-height: 100%;
            padding: 0.7rem 0.75rem;
            box-sizing: border-box;
            position: relative;
        }
        .graph-preview-node-number {
            display: inline-grid;
            place-items: center;
            width: 1.45rem;
            height: 1.45rem;
            margin-bottom: 0.45rem;
            border: 2px solid var(--preview-accent);
            border-radius: 50%;
            background: var(--preview-accent-soft);
            color: var(--preview-text);
            font-size: 0.72rem;
            font-weight: 800;
        }
        .graph-preview-connector {
            align-self: center;
            flex: 0 0 clamp(34px, 3.5vw, 58px);
            height: 2px;
            margin: 0 -1px;
            background:
                linear-gradient(
                    90deg,
                    color-mix(in srgb, var(--preview-accent) 32%, var(--preview-border)),
                    var(--preview-accent)
                );
            font-size: 0;
            position: relative;
            z-index: 2;
        }
        .graph-preview-connector::before {
            content: "";
            position: absolute;
            left: -3px;
            top: -3px;
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: var(--preview-accent);
        }
        .graph-preview-connector::after {
            content: "";
            position: absolute;
            right: -5px;
            top: -5px;
            border-left: 11px solid var(--preview-accent);
            border-top: 6px solid transparent;
            border-bottom: 6px solid transparent;
            filter: drop-shadow(0 0 1px var(--preview-bg));
        }
        .graph-preview-row-bridge {
            height: 58px;
            margin: -0.15rem 0;
            position: relative;
        }
        .graph-preview-bridge-exit,
        .graph-preview-bridge-run,
        .graph-preview-bridge-entry {
            position: absolute;
            background: var(--preview-accent);
        }
        .graph-preview-bridge-exit {
            height: 24px;
            right: clamp(0.4rem, 1.5vw, 1.1rem);
            top: 0;
            width: 2px;
        }
        .graph-preview-bridge-exit::before {
            content: "";
            position: absolute;
            left: -3px;
            top: -3px;
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: var(--preview-accent);
        }
        .graph-preview-bridge-run {
            height: 2px;
            left: clamp(0.4rem, 1.5vw, 1.1rem);
            right: clamp(0.4rem, 1.5vw, 1.1rem);
            top: 24px;
        }
        .graph-preview-bridge-entry {
            height: 28px;
            left: clamp(0.4rem, 1.5vw, 1.1rem);
            top: 24px;
            width: 2px;
        }
        .graph-preview-bridge-entry::after {
            content: "";
            position: absolute;
            bottom: -9px;
            left: 50%;
            transform: translateX(-50%);
            border-left: 6px solid transparent;
            border-right: 6px solid transparent;
            border-top: 10px solid var(--preview-accent);
            filter: drop-shadow(0 0 1px var(--preview-bg));
        }
        .graph-preview-kicker {
            font-size: 0.68rem;
            font-weight: 700;
            letter-spacing: 0.04em;
            margin-bottom: 0.3rem;
            opacity: 0.72;
            text-transform: uppercase;
            color: var(--preview-muted);
        }
        .graph-preview-title {
            font-size: 0.95rem;
            font-weight: 700;
            line-height: 1.25;
            margin-bottom: 0.45rem;
            white-space: normal;
            color: var(--preview-text);
        }
        .graph-preview-meta {
            display: grid;
            gap: 0.28rem;
            font-size: 0.78rem;
            color: var(--preview-muted);
        }
        .graph-preview-meta span {
            display: block;
            overflow-wrap: anywhere;
        }
        .graph-preview-personality {
            margin-top: 0.45rem;
            padding-top: 0.45rem;
            border-top: 1px solid var(--preview-border);
            font-size: 0.78rem;
            overflow-wrap: anywhere;
            color: var(--preview-text);
        }
        @media (max-width: 760px) {
            .graph-preview-row {
                display: grid;
                gap: 0.65rem;
            }
            .graph-preview-connector {
                width: 2px;
                height: 34px;
                justify-self: center;
                background:
                    linear-gradient(
                        180deg,
                        var(--preview-border),
                        var(--preview-accent)
                    );
            }
            .graph-preview-connector::before {
                left: -3px;
                top: -3px;
            }
            .graph-preview-connector::after {
                right: auto;
                top: auto;
                bottom: -8px;
                left: 50%;
                transform: translateX(-50%);
                border-left: 6px solid transparent;
                border-right: 6px solid transparent;
                border-top: 9px solid var(--preview-accent);
                border-bottom: 0;
            }
            .graph-preview-row-bridge {
                height: 42px;
                margin: -0.1rem 0;
            }
            .graph-preview-bridge-exit,
            .graph-preview-bridge-run {
                display: none;
            }
            .graph-preview-bridge-entry {
                height: 34px;
                left: 50%;
                top: 0;
            }
        }
        </style>
        <div class="graph-preview-shell">
            <div class="graph-preview-chain">
        """,
    ]
    total_nodes = len(node_forms)
    nodes_per_row = 4
    for row_start in range(0, total_nodes, nodes_per_row):
        row_node_forms = node_forms[row_start : row_start + nodes_per_row]
        preview_parts.append('<div class="graph-preview-row">')
        for row_position, node_form in enumerate(row_node_forms):
            index = row_start + row_position + 1
            is_preset_personality = node_form.get("personality_mode") == "preset"
            personality_label = node_form.get("personality_preset", "Custom")
            if not is_preset_personality:
                personality_label = node_form.get("personality_custom", "").strip()
                personality_label = personality_label[:140] or "Custom personality"
            node_label = node_form.get("label", "").strip() or "Untitled node"
            node_id = node_form.get("node_id", "").strip() or "missing"
            provider = node_form.get("provider", "").strip() or "missing"
            model = node_form.get("model", "").strip() or "missing"

            if row_position > 0:
                preview_parts.append('<div class="graph-preview-connector"></div>')
            preview_parts.append(
                f"""
                    <div class="graph-preview-unit">
                        <div class="graph-preview-node">
                            <div class="graph-preview-node-number">{index}</div>
                            <div class="graph-preview-kicker">Node {index} of {total_nodes}</div>
                            <div class="graph-preview-title">{escape(node_label)}</div>
                            <div class="graph-preview-meta">
                                <span><strong>ID:</strong> {escape(node_id)}</span>
                                <span><strong>Provider:</strong> {escape(provider)}</span>
                                <span><strong>Model:</strong> {escape(model)}</span>
                            </div>
                            <div class="graph-preview-personality">
                                <strong>Characteristic:</strong> {escape(personality_label)}
                            </div>
                        </div>
                    </div>
                """
            )
        preview_parts.append("</div>")
        if row_start + nodes_per_row < total_nodes:
            preview_parts.append(
                """
                <div class="graph-preview-row-bridge" aria-hidden="true">
                    <div class="graph-preview-bridge-exit"></div>
                    <div class="graph-preview-bridge-run"></div>
                    <div class="graph-preview-bridge-entry"></div>
                </div>
                """
            )

    preview_parts.append(
        """
            </div>
        </div>
        """
    )
    st.html("".join(preview_parts))
