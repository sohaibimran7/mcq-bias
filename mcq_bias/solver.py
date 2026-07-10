"""Solver for the MCQ cue-bias tasks: single generation, except for multi-turn
biases (are_you_sure), where challenge turns are injected between on-policy
generations."""

from inspect_ai.model import ChatMessageAssistant, ChatMessageUser
from inspect_ai.solver import Generate, Solver, TaskState, solver


@solver
def multi_turn_generate() -> Solver:
    """Handle multi-turn biases with on-policy generation.

    For samples with ``followup_user_messages`` in metadata (are_you_sure),
    generates the model's response at each assistant turn rather than using
    pre-filled canned responses. For single-turn samples, behaves identically
    to ``generate()``.
    """

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        followups = state.metadata.get("followup_user_messages")
        if not followups:
            # Single-turn (or post_hoc with pre-filled assistant turn) — one generation.
            return await generate(state)

        # Multi-turn (are_you_sure): force correct answer → inject challenge → generate.
        # The first assistant turn is teacher-forced with the correct ground truth
        # letter (no CoT); the bias is "anything but your first answer", so an
        # on-policy correct first answer and a forced one are equivalent — forcing
        # saves a generation.
        first_answer = state.target.text
        state.messages.append(ChatMessageAssistant(content=first_answer))

        # Inject challenge messages and generate responses.
        # reasoning_history="none" strips reasoning from prior assistant turns so
        # thinking tokens don't leak into subsequent prompts.
        for followup in followups:
            state.messages.append(ChatMessageUser(content=followup))
            state = await generate(state, reasoning_history="none")

        return state

    return solve
