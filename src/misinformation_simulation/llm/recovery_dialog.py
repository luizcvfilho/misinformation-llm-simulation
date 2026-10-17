"""Local interaction for scripts and notebook kernels with a desktop display."""

from __future__ import annotations

import sys

from misinformation_simulation.llm.recovery import (
    RecoveryDecision,
    RecoveryRequest,
)


def desktop_decision(request: RecoveryRequest) -> RecoveryDecision:
    try:
        import tkinter as tk
        from tkinter import ttk

        root = tk.Tk()
    except Exception as error:
        # A remote notebook may have no desktop display or terminal input.
        if sys.stdin is None or not sys.stdin.isatty():
            raise RuntimeError(
                "No recovery display is available. Run with a desktop display or use "
                "RecoverySession with a custom handler."
            ) from error
        return terminal_decision(request)
    decision = RecoveryDecision("cancel")
    root.title("Execution paused")
    root.geometry("760x480")
    root.attributes("-topmost", True)
    ttk.Label(root, text=f"{request.context.provider} / {request.context.model}", padding=12).pack(
        anchor="w"
    )
    text = tk.Text(root, wrap="word", height=16)
    text.insert("1.0", request.message)
    text.configure(state="disabled")
    text.pack(fill="both", expand=True, padx=12)
    ttk.Label(
        root, text="Continuing records a failure and may block dependent steps.", padding=12
    ).pack(anchor="w")
    selected = tk.StringVar(value=request.context.credential)
    if request.context.choices:
        ttk.Label(root, text="First key on retry:").pack(anchor="w", padx=12)
        ttk.Combobox(
            root, textvariable=selected, values=request.context.choices, state="readonly"
        ).pack(anchor="w", padx=12)

    def choose(action):
        nonlocal decision
        decision = RecoveryDecision(action, selected.get())
        root.destroy()

    buttons = ttk.Frame(root, padding=12)
    buttons.pack(fill="x")
    ttk.Button(buttons, text="Try again", command=lambda: choose("retry")).pack(side="left")
    ttk.Button(buttons, text="Skip this occurrence", command=lambda: choose("skip")).pack(
        side="left", padx=8
    )
    ttk.Button(
        buttons,
        text="Allow this error in future executions and continue",
        command=lambda: choose("allow"),
    ).pack(side="left", padx=8)
    root.protocol("WM_DELETE_WINDOW", root.destroy)
    root.mainloop()
    return decision


def terminal_decision(request: RecoveryRequest) -> RecoveryDecision:
    print(f"\nExecution paused ({request.context.provider}):\n{request.message}")
    while True:
        action = input("[r] Try again / [s] Skip once / [a] Allow in future / [c] Cancel: ").lower()
        if action == "s":
            return RecoveryDecision("skip")
        if action == "a":
            return RecoveryDecision("allow")
        if action == "c":
            return RecoveryDecision("cancel")
        if action == "r":
            slot = request.context.credential
            if len(request.context.choices) > 1:
                slot = input(f"First key [{' / '.join(request.context.choices)}]: ").strip().lower()
                if slot not in request.context.choices:
                    continue
            return RecoveryDecision("retry", slot)
