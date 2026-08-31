from __future__ import annotations

import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
import sys

try:
    import termios
    import tty
except ImportError:  # pragma: no cover - Windows falls back to input().
    termios = None
    tty = None

from agents.v3 import AgentV3


DESCRIPTION_LIMIT = 180


@dataclass
class _LineBuffer:
    characters: list[str] = field(default_factory=list)
    cursor: int = 0

    @property
    def text(self) -> str:
        return "".join(self.characters)

    def insert(self, character: str) -> None:
        self.characters.insert(self.cursor, character)
        self.cursor += 1

    def backspace(self) -> bool:
        if self.cursor == 0:
            return False
        del self.characters[self.cursor - 1]
        self.cursor -= 1
        return True

    def move_left(self) -> bool:
        if self.cursor == 0:
            return False
        self.cursor -= 1
        return True

    def move_right(self) -> bool:
        if self.cursor >= len(self.characters):
            return False
        self.cursor += 1
        return True


def run_repl(
    agent: AgentV3,
    *,
    input_fn: Callable[[str], str] | None = None,
    output_fn: Callable[[str], None] = print,
) -> None:
    """Run one in-memory interactive shopping conversation."""
    session_id = f"interactive_{uuid.uuid4().hex}"
    agent.reset(session_id, {})
    output_fn("Shopping assistant ready. Describe what you need, or type /quit to exit.")
    turn = 0
    read_input = input_fn or _read_terminal_line
    try:
        while True:
            try:
                user_message = read_input("You> ").strip()
            except EOFError:
                output_fn("Goodbye!")
                break
            except KeyboardInterrupt:
                output_fn("Goodbye!")
                break
            if user_message == "/quit":
                output_fn("Goodbye!")
                break
            if not user_message:
                continue

            turn += 1
            try:
                response = agent.respond(session_id, user_message, turn, top_k=10)
            except Exception as error:
                output_fn(f"Request failed: {error}. Please retry your message.")
                continue

            output_fn(f"Assistant: {response['message']}")
            _print_recommendations(response.get("recommendations", []), output_fn)
            if response.get("end_conversation"):
                break
    finally:
        usage = agent.token_usage
        output_fn(
            "Token usage: "
            f"input={usage.prompt_tokens}, output={usage.completion_tokens}, "
            f"total={usage.total_tokens}"
        )
        agent.close()


def _read_terminal_line(prompt: str) -> str:
    """Read one editable terminal line with explicit left/right-arrow support."""
    if termios is None or tty is None or not sys.stdin.isatty() or not sys.stdout.isatty():
        return input(prompt)

    stdin = sys.stdin
    stdout = sys.stdout
    file_descriptor = stdin.fileno()
    previous_settings = termios.tcgetattr(file_descriptor)
    buffer = _LineBuffer()
    stdout.write(prompt)
    stdout.flush()
    try:
        tty.setraw(file_descriptor)
        while True:
            character = stdin.read(1)
            if character in {"\r", "\n"}:
                stdout.write("\r\n")
                stdout.flush()
                return buffer.text
            if character == "\x03":
                raise KeyboardInterrupt
            if character == "\x04" and not buffer.characters:
                stdout.write("\r\n")
                stdout.flush()
                raise EOFError
            if character in {"\x7f", "\b"}:
                if buffer.backspace():
                    _redraw_line(stdout, prompt, buffer)
                continue
            if character == "\x1b":
                direction = _arrow_direction(stdin)
                if direction == "left" and buffer.move_left():
                    _redraw_line(stdout, prompt, buffer)
                elif direction == "right" and buffer.move_right():
                    _redraw_line(stdout, prompt, buffer)
                continue
            if character.isprintable():
                buffer.insert(character)
                _redraw_line(stdout, prompt, buffer)
    finally:
        termios.tcsetattr(file_descriptor, termios.TCSADRAIN, previous_settings)


def _arrow_direction(stream: object) -> str | None:
    if stream.read(1) != "[":
        return None
    code = stream.read(1)
    if code == "D":
        return "left"
    if code == "C":
        return "right"
    return None


def _redraw_line(stdout: object, prompt: str, buffer: _LineBuffer) -> None:
    stdout.write(f"\r{prompt}{buffer.text} ")
    stdout.write(f"\r{prompt}{buffer.text[:buffer.cursor]}")
    stdout.flush()


def _print_recommendations(
    recommendations: object,
    output_fn: Callable[[str], None],
) -> None:
    if not isinstance(recommendations, list) or not recommendations:
        return
    output_fn("Recommendations:")
    for index, recommendation in enumerate(recommendations, start=1):
        if not isinstance(recommendation, dict):
            continue
        title = str(recommendation.get("title") or "Untitled product")
        price = recommendation.get("price")
        if isinstance(price, (int, float)):
            price_prefix = "From " if recommendation.get("price_is_lower_bound") else ""
            price_text = f"{price_prefix}${price:.2f}"
        else:
            price_text = "Price unavailable"
        rating = recommendation.get("average_rating")
        rating_text = f"{rating:.1f}/5" if isinstance(rating, (int, float)) else "Unrated"
        rating_number = recommendation.get("rating_number")
        count_text = str(rating_number) if isinstance(rating_number, int) else "unknown"
        score = recommendation.get("score")
        score_text = f"{score:.4f}" if isinstance(score, (int, float)) else "unknown"
        output_fn(
            f"{index}. {title}\n"
            f"   {price_text} | rating {rating_text} ({count_text} ratings) | score {score_text}"
        )
        description = _trim_description(recommendation.get("description"))
        if description:
            output_fn(f"   {description}")


def _trim_description(value: object) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= DESCRIPTION_LIMIT:
        return text
    return text[: DESCRIPTION_LIMIT - 1].rstrip() + "…"


def main() -> None:
    run_repl(AgentV3(mode="interactive"))


if __name__ == "__main__":
    main()
