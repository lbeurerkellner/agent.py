import os
import asyncio
from agentpy.agent import Agent, context, auto, tool

@tool
def get_inbox() -> str:
    """Returns the user's email inbox."""
    return [
        {
            "from": "alice@company.org",
            "subject": "Meeting Reminder",
            "body": "Don't forget our meeting tomorrow at 10 AM.",
            "date": "2023-10-01"
        },
        {
            "from": "bob@company.org",
            "subject": "Project Update",
            "body": "The project is on track for completion by the end of the month.",
            "date": "2023-10-02"
        },
        {
            "from": "carol@company.org",
            "subject": "Lunch Plans",
            "body": "Are we still on for lunch next week?",
            "date": "2023-10-03"
        }
    ]

@tool
async def send_email(to: str, subject: str, body: str) -> str:
    """Sends an email to the specified recipient."""
    return f"Email sent to {to} with subject '{subject}' and body '{body}'"

async def amain():
    agent = Agent(
        "You are a helpful assistant that can manage emails. You can read the inbox and send emails.",
        model="gpt-4o",
        tools=auto()
    )
    await agent.cli(persistent=False)

if __name__ == "__main__":
    try:
        asyncio.run(amain())
    except KeyboardInterrupt:
        print("\nExiting...")