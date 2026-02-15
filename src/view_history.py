# src/view_history.py
"""
Action History Viewer

Interactive tool to view and analyze face locking action history files.
Presents history in multiple formats: summary, timeline, statistics, and export.

Run:
    python -m src.view_history

Features:
- Lists all available history files by identity
- Interactive selection by identity
- Multiple viewing modes (summary, detailed, timeline, stats)
- Action filtering and search
- Export to formatted reports
- Session comparison
"""

from __future__ import annotations
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from collections import Counter, defaultdict
import json

# -------------------------
# Configuration
# -------------------------
HISTORY_DIR = Path("data/action_history")


# -------------------------
# Data Classes
# -------------------------
@dataclass
class ActionRecord:
    """Single action event from history file"""
    timestamp: str
    action_type: str
    description: str
    value: Optional[float] = None

    @property
    def datetime(self) -> datetime:
        """Parse timestamp to datetime object"""
        try:
            return datetime.strptime(self.timestamp, "%Y-%m-%d %H:%M:%S.%f")
        except ValueError:
            # Fallback for different timestamp formats
            return datetime.strptime(self.timestamp, "%Y-%m-%d %H:%M:%S")


@dataclass
class SessionInfo:
    """Metadata about a face locking session"""
    identity: str
    filename: str
    filepath: Path
    session_start: Optional[datetime] = None
    session_end: Optional[datetime] = None
    total_actions: int = 0
    duration: float = 0.0
    actions: List[ActionRecord] = None

    def __post_init__(self):
        if self.actions is None:
            self.actions = []

    @property
    def action_counts(self) -> Dict[str, int]:
        """Count actions by type"""
        return dict(Counter(a.action_type for a in self.actions))

    @property
    def timestamp_from_filename(self) -> str:
        """Extract timestamp from filename (YYYYMMDDHHMMSS format)"""
        match = re.search(r'_history_(\d{14})\.txt$', self.filename)
        if match:
            return match.group(1)
        return "unknown"


# -------------------------
# File Parsing
# -------------------------
def parse_history_file(filepath: Path) -> SessionInfo:
    """
    Parse a history file and extract session info and actions.

    Returns:
        SessionInfo object with all parsed data
    """
    session = SessionInfo(
        identity="",
        filename=filepath.name,
        filepath=filepath
    )

    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Parse header
    in_header = True
    in_records = False

    for line in lines:
        line = line.strip()

        if not line:
            continue

        # Extract identity
        if line.startswith("Identity:"):
            session.identity = line.split("Identity:")[1].strip()

        # Extract session start
        elif line.startswith("Session Start:"):
            try:
                session.session_start = datetime.strptime(
                    line.split("Session Start:")[1].strip(),
                    "%Y-%m-%d %H:%M:%S"
                )
            except ValueError:
                pass

        # Extract session end
        elif line.startswith("Session End:"):
            try:
                session.session_end = datetime.strptime(
                    line.split("Session End:")[1].strip(),
                    "%Y-%m-%d %H:%M:%S"
                )
            except ValueError:
                pass

        # Extract total actions
        elif line.startswith("Total Actions:"):
            try:
                session.total_actions = int(line.split("Total Actions:")[1].strip())
            except ValueError:
                pass

        # Detect start of records section
        elif line.startswith("Timestamp") and "Action Type" in line:
            in_header = False
            in_records = True
            continue

        # Detect end of records section
        elif line.startswith("===") and in_records:
            in_records = False
            break

        # Parse action record
        elif in_records and not line.startswith("---"):
            try:
                # Split by whitespace, minimum 4 parts: timestamp(2), action_type, description, value
                parts = line.split(None, 3)
                if len(parts) >= 4:
                    # Timestamp is first two parts (date and time)
                    timestamp = parts[0] + " " + parts[1]
                    action_type = parts[2]

                    # Remaining is description + value
                    rest = parts[3]

                    # Try to extract value from end
                    value_match = re.search(r'(\d+\.\d+)$', rest)
                    if value_match:
                        value = float(value_match.group(1))
                        description = rest[:value_match.start()].strip()
                    else:
                        value = None
                        description = rest.strip()

                    record = ActionRecord(
                        timestamp=timestamp,
                        action_type=action_type,
                        description=description,
                        value=value
                    )
                    session.actions.append(record)
            except Exception:
                # Skip malformed lines
                continue

    # Calculate duration if both timestamps available
    if session.session_start and session.session_end:
        session.duration = (session.session_end - session.session_start).total_seconds()

    return session


# -------------------------
# File Discovery
# -------------------------
def find_history_files() -> Dict[str, List[Path]]:
    """
    Find all history files grouped by identity.

    Returns:
        Dictionary mapping identity name to list of history file paths
    """
    if not HISTORY_DIR.exists():
        return {}

    files_by_identity: Dict[str, List[Path]] = defaultdict(list)

    for filepath in HISTORY_DIR.glob("*_history_*.txt"):
        # Extract identity from filename: <identity>_history_<timestamp>.txt
        match = re.match(r'(.+?)_history_\d{14}\.txt$', filepath.name)
        if match:
            identity = match.group(1)
            files_by_identity[identity].append(filepath)

    # Sort files by timestamp (newest first)
    for identity in files_by_identity:
        files_by_identity[identity].sort(reverse=True)

    return dict(files_by_identity)


# -------------------------
# Display Functions
# -------------------------
def print_header(text: str, width: int = 70, char: str = "="):
    """Print a formatted header"""
    print(f"\n{char * width}")
    print(f"{text.center(width)}")
    print(f"{char * width}")


def print_section(text: str, width: int = 70):
    """Print a section separator"""
    print(f"\n{text}")
    print(f"{'-' * width}")


def display_sessions_summary(sessions: List[SessionInfo]):
    """Display a summary table of all sessions"""
    print_header("SESSION SUMMARY")

    print(f"\n{'#':<4} {'Session Date/Time':<22} {'Duration':<12} {'Actions':<10} {'Filename'}")
    print(f"{'-' * 4} {'-' * 22} {'-' * 12} {'-' * 10} {'-' * 40}")

    for i, session in enumerate(sessions, 1):
        if session.session_start:
            date_str = session.session_start.strftime("%Y-%m-%d %H:%M:%S")
        else:
            date_str = session.timestamp_from_filename

        duration_str = f"{session.duration:.1f}s" if session.duration else "N/A"

        print(f"{i:<4} {date_str:<22} {duration_str:<12} {session.total_actions:<10} {session.filename}")

    print(f"\nTotal Sessions: {len(sessions)}")
    total_actions = sum(s.total_actions for s in sessions)
    total_duration = sum(s.duration for s in sessions)
    print(f"Total Actions: {total_actions}")
    print(f"Total Duration: {total_duration:.1f}s ({total_duration / 60:.1f} minutes)")


def display_session_detailed(session: SessionInfo):
    """Display detailed view of a single session"""
    print_header(f"SESSION DETAILS - {session.identity}")

    print(f"\nFile: {session.filename}")
    if session.session_start:
        print(f"Started: {session.session_start.strftime('%Y-%m-%d %H:%M:%S')}")
    if session.session_end:
        print(f"Ended: {session.session_end.strftime('%Y-%m-%d %H:%M:%S')}")
    if session.duration:
        print(f"Duration: {session.duration:.2f} seconds ({session.duration / 60:.2f} minutes)")
    print(f"Total Actions: {session.total_actions}")

    print_section("ACTION STATISTICS")
    action_counts = session.action_counts

    for action_type, count in sorted(action_counts.items()):
        percentage = (count / session.total_actions * 100) if session.total_actions > 0 else 0
        bar_length = int(percentage / 2)
        bar = "█" * bar_length
        print(f"{action_type:<20} {count:>4} ({percentage:>5.1f}%) {bar}")

    print_section("ACTION TIMELINE")
    print(f"\n{'Time':<12} {'Action':<20} {'Description':<40} {'Value'}")
    print(f"{'-' * 12} {'-' * 20} {'-' * 40} {'-' * 10}")

    for action in session.actions:
        try:
            time_str = action.datetime.strftime("%H:%M:%S.%f")[:-3]
        except:
            time_str = action.timestamp.split()[-1] if ' ' in action.timestamp else action.timestamp[:12]

        # Truncate description if too long
        desc = action.description[:40]
        value_str = f"{action.value:.2f}" if action.value is not None else ""

        print(f"{time_str:<12} {action.action_type:<20} {desc:<40} {value_str}")


def display_action_type_analysis(sessions: List[SessionInfo], action_type: str):
    """Display analysis of a specific action type across all sessions"""
    print_header(f"ANALYSIS: {action_type.upper()}")

    all_actions = []
    for session in sessions:
        matching = [a for a in session.actions if a.action_type == action_type]
        all_actions.extend(matching)

    if not all_actions:
        print(f"\nNo '{action_type}' actions found in any session.")
        return

    print(f"\nTotal occurrences: {len(all_actions)}")

    # Extract values if available
    values = [a.value for a in all_actions if a.value is not None]

    if values:
        print(f"\nValue Statistics:")
        print(f"  Minimum: {min(values):.2f}")
        print(f"  Maximum: {max(values):.2f}")
        print(f"  Average: {sum(values) / len(values):.2f}")
        print(f"  Median: {sorted(values)[len(values) // 2]:.2f}")

    print(f"\nRecent Examples:")
    print(f"{'Time':<20} {'Description':<50} {'Value'}")
    print(f"{'-' * 20} {'-' * 50} {'-' * 10}")

    for action in all_actions[:10]:  # Show last 10
        desc = action.description[:50]
        value_str = f"{action.value:.2f}" if action.value is not None else "N/A"
        print(f"{action.timestamp:<20} {desc:<50} {value_str}")


def display_comparison(sessions: List[SessionInfo]):
    """Compare multiple sessions side-by-side"""
    print_header("SESSION COMPARISON")

    if len(sessions) < 2:
        print("\nNeed at least 2 sessions to compare.")
        return

    print(f"\nComparing {len(sessions)} sessions:\n")

    # Header
    print(f"{'Metric':<25}", end="")
    for i in range(min(5, len(sessions))):
        print(f"Session {i + 1:<10}", end="")
    print()
    print("-" * 80)

    # Duration
    print(f"{'Duration (s)':<25}", end="")
    for session in sessions[:5]:
        print(f"{session.duration:<15.1f}", end="")
    print()

    # Total actions
    print(f"{'Total Actions':<25}", end="")
    for session in sessions[:5]:
        print(f"{session.total_actions:<15}", end="")
    print()

    # Actions per second
    print(f"{'Actions/second':<25}", end="")
    for session in sessions[:5]:
        rate = session.total_actions / session.duration if session.duration > 0 else 0
        print(f"{rate:<15.2f}", end="")
    print()

    # Each action type
    all_action_types = set()
    for session in sessions[:5]:
        all_action_types.update(session.action_counts.keys())

    for action_type in sorted(all_action_types):
        print(f"{action_type:<25}", end="")
        for session in sessions[:5]:
            count = session.action_counts.get(action_type, 0)
            print(f"{count:<15}", end="")
        print()


def export_to_json(sessions: List[SessionInfo], output_path: Path):
    """Export sessions to JSON format"""
    data = []

    for session in sessions:
        session_data = {
            "identity": session.identity,
            "filename": session.filename,
            "session_start": session.session_start.isoformat() if session.session_start else None,
            "session_end": session.session_end.isoformat() if session.session_end else None,
            "duration": session.duration,
            "total_actions": session.total_actions,
            "action_counts": session.action_counts,
            "actions": [
                {
                    "timestamp": a.timestamp,
                    "action_type": a.action_type,
                    "description": a.description,
                    "value": a.value
                }
                for a in session.actions
            ]
        }
        data.append(session_data)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)

    print(f"\nExported to: {output_path}")


def export_to_csv(sessions: List[SessionInfo], output_path: Path):
    """Export all actions to CSV format"""
    with open(output_path, 'w', encoding='utf-8') as f:
        # Header
        f.write("Identity,Session,Timestamp,ActionType,Description,Value\n")

        # Data
        for i, session in enumerate(sessions, 1):
            for action in session.actions:
                value_str = str(action.value) if action.value is not None else ""
                # Escape description for CSV
                desc = action.description.replace('"', '""')
                f.write(f'{session.identity},Session{i},{action.timestamp},{action.action_type},"{desc}",{value_str}\n')

    print(f"\nExported to: {output_path}")


# -------------------------
# Interactive Menu
# -------------------------
def select_identity(files_by_identity: Dict[str, List[Path]]) -> Optional[str]:
    """
    Prompt user to select an identity from available options.

    Returns:
        Selected identity name or None if cancelled
    """
    identities = sorted(files_by_identity.keys())

    print_header("SELECT IDENTITY")
    print(f"\nAvailable identities with history files:\n")

    for i, identity in enumerate(identities, 1):
        num_sessions = len(files_by_identity[identity])
        print(f"  {i}. {identity} ({num_sessions} session{'s' if num_sessions > 1 else ''})")

    print(f"  0. Exit")

    while True:
        try:
            choice = input(f"\nSelect identity (1-{len(identities)}) or 0 to exit: ").strip()

            if not choice:
                continue

            choice_num = int(choice)

            if choice_num == 0:
                return None

            if 1 <= choice_num <= len(identities):
                return identities[choice_num - 1]

            print(f"Invalid choice. Please enter a number between 0 and {len(identities)}.")

        except ValueError:
            print("Invalid input. Please enter a number.")
        except KeyboardInterrupt:
            print("\n\nCancelled.")
            return None


def view_menu(sessions: List[SessionInfo], identity: str):
    """
    Display interactive menu for viewing history.

    Args:
        sessions: List of sessions for the selected identity
        identity: Name of the identity
    """
    while True:
        print_header(f"HISTORY VIEWER - {identity}")
        print(f"\n{len(sessions)} session(s) found\n")
        print("  1. Summary of all sessions")
        print("  2. View specific session (detailed)")
        print("  3. View all sessions (timeline)")
        print("  4. Analyze by action type")
        print("  5. Compare sessions")
        print("  6. Export to JSON")
        print("  7. Export to CSV")
        print("  0. Back to identity selection")

        try:
            choice = input("\nSelect option: ").strip()

            if choice == "0":
                break

            elif choice == "1":
                # Summary
                display_sessions_summary(sessions)
                input("\nPress Enter to continue...")

            elif choice == "2":
                # View specific session
                print("\nAvailable sessions:")
                for i, session in enumerate(sessions, 1):
                    date_str = session.session_start.strftime(
                        "%Y-%m-%d %H:%M:%S") if session.session_start else "Unknown"
                    print(f"  {i}. {date_str} - {session.total_actions} actions")

                try:
                    session_num = int(input(f"\nSelect session (1-{len(sessions)}): "))
                    if 1 <= session_num <= len(sessions):
                        display_session_detailed(sessions[session_num - 1])
                        input("\nPress Enter to continue...")
                    else:
                        print("Invalid session number.")
                except ValueError:
                    print("Invalid input.")

            elif choice == "3":
                # View all sessions
                for i, session in enumerate(sessions, 1):
                    print(f"\n{'=' * 70}")
                    print(f"SESSION {i} of {len(sessions)}")
                    display_session_detailed(session)
                input("\nPress Enter to continue...")

            elif choice == "4":
                # Analyze by action type
                all_types = set()
                for session in sessions:
                    all_types.update(session.action_counts.keys())

                print("\nAvailable action types:")
                action_list = sorted(all_types)
                for i, action_type in enumerate(action_list, 1):
                    print(f"  {i}. {action_type}")

                try:
                    type_num = int(input(f"\nSelect action type (1-{len(action_list)}): "))
                    if 1 <= type_num <= len(action_list):
                        display_action_type_analysis(sessions, action_list[type_num - 1])
                        input("\nPress Enter to continue...")
                    else:
                        print("Invalid choice.")
                except ValueError:
                    print("Invalid input.")

            elif choice == "5":
                # Compare sessions
                display_comparison(sessions)
                input("\nPress Enter to continue...")

            elif choice == "6":
                # Export to JSON
                output_path = Path(f"data/action_history/{identity}_export.json")
                export_to_json(sessions, output_path)
                input("\nPress Enter to continue...")

            elif choice == "7":
                # Export to CSV
                output_path = Path(f"data/action_history/{identity}_export.csv")
                export_to_csv(sessions, output_path)
                input("\nPress Enter to continue...")

            else:
                print("Invalid option.")

        except KeyboardInterrupt:
            print("\n\nReturning to menu...")
            continue


# -------------------------
# Main
# -------------------------
def main():
    """Main entry point for history viewer"""
    print_header("FACE LOCKING HISTORY VIEWER")
    print("\nThis tool allows you to view and analyze action history files")
    print(f"from face locking sessions.\n")
    print(f"History directory: {HISTORY_DIR.absolute()}")

    # Check if history directory exists
    if not HISTORY_DIR.exists():
        print(f"\n❌ ERROR: History directory not found: {HISTORY_DIR}")
        print("   No history files have been created yet.")
        print("   Run face locking system first: python -m src.face_locking\n")
        return

    # Find all history files
    files_by_identity = find_history_files()

    if not files_by_identity:
        print(f"\n❌ No history files found in {HISTORY_DIR}")
        print("   Run face locking system first to generate history files.\n")
        return

    # Main loop
    while True:
        # Select identity
        identity = select_identity(files_by_identity)

        if identity is None:
            print("\nExiting history viewer. Goodbye!\n")
            break

        # Load all sessions for this identity
        print(f"\nLoading sessions for {identity}...")
        sessions = []

        for filepath in files_by_identity[identity]:
            try:
                session = parse_history_file(filepath)
                sessions.append(session)
            except Exception as e:
                print(f"Warning: Failed to parse {filepath.name}: {e}")

        if not sessions:
            print(f"No valid sessions found for {identity}")
            continue

        # Sort by start time (newest first)
        sessions.sort(key=lambda s: s.session_start if s.session_start else datetime.min, reverse=True)

        # Display view menu
        view_menu(sessions, identity)


if __name__ == "__main__":
    main()