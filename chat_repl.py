
#!/usr/bin/env python3
# Minimal REPL for VERA–ORUS–OROS

import argparse, json
from prototype import TriadRunner

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", default="repl")
    args = ap.parse_args()

    runner = TriadRunner()
    print(f"[triad] session={args.session}")
    print("Type your message. Ctrl+C to exit.\n")

    while True:
        try:
            msg = input("you> ").strip()
            if not msg:
                continue
            out = runner.run(msg, session_id=args.session)
            print("\n--- result ---")
            print("approved:", out.get("approved"))
            print("verum_score:", out.get("metrics", {}).get("verum_score"))
            print("passes:", out.get("passes"))
            print("fails:", out.get("fails"))
            print("sources:", out.get("sources"))
            print("uncertainty_note:", out.get("uncertainty_note"))
            print("limits:", out.get("limits"))
            print("\nfinal_text:\n")
            print(out.get("final_text", ""))
            print("--------------\n")
        except (KeyboardInterrupt, EOFError):
            print("\n[bye]")
            break

if __name__ == "__main__":
    main()
