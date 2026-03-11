import time
import sys
import threading

# ── ANSI colour / style codes ──────────────────────────────────────────────
RESET   = "\033[0m"
BOLD    = "\033[1m"
DIM     = "\033[2m"

BLACK   = "\033[30m"
WHITE   = "\033[97m"
CYAN    = "\033[96m"
MAGENTA = "\033[95m"
YELLOW  = "\033[93m"
GREEN   = "\033[92m"
RED     = "\033[91m"
BLUE    = "\033[94m"

BG_BLACK = "\033[40m"


def clear_line():
    sys.stdout.write("\r\033[K")
    sys.stdout.flush()


def print_centered(text, width=60):
    print(text.center(width))


def banner():
    """Print the fancy title banner."""
    w = 62

    top    = "╔" + "═" * (w - 2) + "╗"
    bottom = "╚" + "═" * (w - 2) + "╝"
    mid    = "║" + " " * (w - 2) + "║"

    accent_bar = (
        CYAN + "▓▓▓" + MAGENTA + "▓▓▓" + YELLOW + "▓▓▓" +
        GREEN + "▓▓▓" + BLUE + "▓▓▓" + RED + "▓▓▓" + RESET
    )

    title_line   = f"  {BOLD}{WHITE}VISUALISATION GENERATOR{RESET}  "
    tagline_line = f"   {DIM}{CYAN}⬡  next-gen rendering engine  ⬡{RESET}   "

    print()
    print(f"{BOLD}{CYAN}{top}{RESET}")
    print(f"{BOLD}{CYAN}║{RESET}" + " " * (w - 2) + f"{BOLD}{CYAN}║{RESET}")

    # accent bar row
    bar_raw = "▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓"
    colors = [CYAN, MAGENTA, YELLOW, GREEN, BLUE, RED, CYAN, MAGENTA, YELLOW, GREEN]
    colored_bar = ""
    for i, ch in enumerate(bar_raw):
        colored_bar += colors[i % len(colors)] + ch
    colored_bar += RESET
    print(f"{BOLD}{CYAN}║{RESET}" + colored_bar + f"{BOLD}{CYAN}║{RESET}")

    print(f"{BOLD}{CYAN}║{RESET}" + " " * (w - 2) + f"{BOLD}{CYAN}║{RESET}")

    # Title – centred inside the box (w-2 visible chars wide)
    inner = w - 2
    visible_title = "VISUALISATION GENERATOR"
    pad = (inner - len(visible_title)) // 2
    title_row = " " * pad + f"{BOLD}{WHITE}{visible_title}{RESET}" + " " * (inner - pad - len(visible_title))
    print(f"{BOLD}{CYAN}║{RESET}" + title_row + f"{BOLD}{CYAN}║{RESET}")

    visible_tag = "⬡  next-gen rendering engine  ⬡"
    pad2 = (inner - len(visible_tag)) // 2
    tag_row = " " * pad2 + f"{DIM}{CYAN}{visible_tag}{RESET}" + " " * (inner - pad2 - len(visible_tag))
    print(f"{BOLD}{CYAN}║{RESET}" + tag_row + f"{BOLD}{CYAN}║{RESET}")

    print(f"{BOLD}{CYAN}║{RESET}" + " " * (w - 2) + f"{BOLD}{CYAN}║{RESET}")
    print(f"{BOLD}{CYAN}{bottom}{RESET}")
    print()


def spinner_task(stop_event, label="Processing"):
    """Animated spinner that runs in a background thread."""
    frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    idx = 0
    while not stop_event.is_set():
        frame = frames[idx % len(frames)]
        sys.stdout.write(f"\r  {CYAN}{BOLD}{frame}{RESET}  {WHITE}{label}{RESET}  {DIM}…{RESET}")
        sys.stdout.flush()
        idx += 1
        time.sleep(0.08)


def progress_bar(duration=10, width=40, label="Generating"):
    """Display a smooth animated progress bar over `duration` seconds."""
    steps      = 200                      # total update steps
    sleep_time = duration / steps

    dots_cycle = ["∙  ", "∙∙ ", "∙∙∙", " ∙∙", "  ∙", "   "]

    for i in range(steps + 1):
        pct      = i / steps
        filled   = int(pct * width)
        empty    = width - filled
        bar_fill = CYAN + "█" * filled + MAGENTA + "▒" * empty + RESET

        # colour the percentage
        if pct < 0.5:
            pct_col = YELLOW
        elif pct < 0.85:
            pct_col = CYAN
        else:
            pct_col = GREEN

        dots = dots_cycle[i % len(dots_cycle)]

        sys.stdout.write(
            f"\r  {WHITE}{label}{RESET} "
            f"[{bar_fill}] "
            f"{pct_col}{BOLD}{pct*100:5.1f}%{RESET} "
            f"{DIM}{dots}{RESET}"
        )
        sys.stdout.flush()
        time.sleep(sleep_time)

    sys.stdout.write("\n")
    sys.stdout.flush()


def eta_countdown(seconds=10):
    """Display ETA countdown ticking down each second."""
    for remaining in range(seconds, 0, -1):
        bar_len  = 20
        elapsed  = seconds - remaining
        filled   = int((elapsed / seconds) * bar_len)
        mini_bar = GREEN + "━" * filled + DIM + "─" * (bar_len - filled) + RESET

        sys.stdout.write(
            f"\r  {DIM}ETA{RESET}  "
            f"{YELLOW}{BOLD}{remaining:>2}s{RESET}  "
            f"[{mini_bar}]  "
            f"{DIM}{CYAN}hold tight…{RESET}   "
        )
        sys.stdout.flush()
        time.sleep(1)

    clear_line()


def section_header(text):
    line = f"{DIM}{CYAN}{'─' * 4}{RESET}  {BOLD}{WHITE}{text}{RESET}  {DIM}{CYAN}{'─' * 4}{RESET}"
    print(f"  {line}")


def success_banner():
    w = 62
    top    = "╔" + "═" * (w - 2) + "╗"
    bottom = "╚" + "═" * (w - 2) + "╝"
    inner  = w - 2

    print()
    print(f"{BOLD}{GREEN}{top}{RESET}")

    msg1 = "✦  VISUALISATION GENERATED  ✦"
    p1   = (inner - len(msg1)) // 2
    row1 = " " * p1 + f"{BOLD}{GREEN}{msg1}{RESET}" + " " * (inner - p1 - len(msg1))
    print(f"{BOLD}{GREEN}║{RESET}" + row1 + f"{BOLD}{GREEN}║{RESET}")

    msg2 = "output ready · 100% complete"
    p2   = (inner - len(msg2)) // 2
    row2 = " " * p2 + f"{DIM}{WHITE}{msg2}{RESET}" + " " * (inner - p2 - len(msg2))
    print(f"{BOLD}{GREEN}║{RESET}" + row2 + f"{BOLD}{GREEN}║{RESET}")

    print(f"{BOLD}{GREEN}{bottom}{RESET}")
    print()


# ── MAIN ───────────────────────────────────────────────────────────────────
def main():
    # 1. Title banner
    banner()
    time.sleep(0.4)

    # 2. ETA notice
    section_header("INITIALISING")
    print(f"  {DIM}Render engine warming up  ·  {YELLOW}{BOLD}ETA 10 seconds{RESET}")
    print()

    # 3. Live countdown (10 × 1 s ticks)
    eta_countdown(10)

    # 4. Progress bar (fills over 10 s)
    section_header("RENDERING")
    print()
    progress_bar(duration=10, width=42, label="Visualisation")
    print()

    # 5. Final success banner
    success_banner()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n\n  {DIM}Interrupted.{RESET}\n")
