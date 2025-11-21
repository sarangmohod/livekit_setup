# import logging

# from dotenv import load_dotenv
# from livekit import rtc
# from livekit.agents import (
#     Agent,
#     AgentServer,
#     AgentSession,
#     JobContext,
#     JobProcess,
#     cli,
#     inference,
#     room_io,
# )
# from livekit.plugins import noise_cancellation, silero
# from livekit.plugins.turn_detector.multilingual import MultilingualModel
# from livekit.plugins import assemblyai

# logger = logging.getLogger("agent")

# load_dotenv(".env.local")


# class Assistant(Agent):
#     def __init__(self) -> None:
#         super().__init__(
#             instructions="""You are a helpful voice AI assistant. The user is interacting with you via voice, even if you perceive the conversation as text.
#             You eagerly assist users with their questions by providing information from your extensive knowledge.
#             Your responses are concise, to the point, and without any complex formatting or punctuation including emojis, asterisks, or other symbols.
#             You are curious, friendly, and have a sense of humor.""",
#         )

#     # To add tools, use the @function_tool decorator.
#     # Here's an example that adds a simple weather tool.
#     # You also have to add `from livekit.agents import function_tool, RunContext` to the top of this file
#     # @function_tool
#     # async def lookup_weather(self, context: RunContext, location: str):
#     #     """Use this tool to look up current weather information in the given location.
#     #
#     #     If the location is not supported by the weather service, the tool will indicate this. You must tell the user the location's weather is unavailable.
#     #
#     #     Args:
#     #         location: The location to look up weather information for (e.g. city name)
#     #     """
#     #
#     #     logger.info(f"Looking up weather for {location}")
#     #
#     #     return "sunny with a temperature of 70 degrees."


# server = AgentServer()


# def prewarm(proc: JobProcess):
#     proc.userdata["vad"] = silero.VAD.load()


# server.setup_fnc = prewarm


# @server.rtc_session()
# async def my_agent(ctx: JobContext):
#     # Logging setup
#     # Add any other context you want in all log entries here
#     ctx.log_context_fields = {
#         "room": ctx.room.name,
#     }

#     # Set up a voice AI pipeline using OpenAI, Cartesia, AssemblyAI, and the LiveKit turn detector
#     session = AgentSession(
#         # Speech-to-text (STT) is your agent's ears, turning the user's speech into text that the LLM can understand
#         # See all available models at https://docs.livekit.io/agents/models/stt/
#         stt = assemblyai.STT(),
#         # VAD and turn detection are used to determine when the user is speaking and when the agent should respond
#         # See more at https://docs.livekit.io/agents/build/turns
#         turn_detection=MultilingualModel(),
#         vad=ctx.proc.userdata["vad"],
#     )

#     # To use a realtime model instead of a voice pipeline, use the following session setup instead.
#     # (Note: This is for the OpenAI Realtime API. For other providers, see https://docs.livekit.io/agents/models/realtime/))
#     # 1. Install livekit-agents[openai]
#     # 2. Set OPENAI_API_KEY in .env.local
#     # 3. Add `from livekit.plugins import openai` to the top of this file
#     # 4. Use the following session setup instead of the version above
#     # session = AgentSession(
#     #     llm=openai.realtime.RealtimeModel(voice="marin")
#     # )

#     # # Add a virtual avatar to the session, if desired
#     # # For other providers, see https://docs.livekit.io/agents/models/avatar/
#     # avatar = hedra.AvatarSession(
#     #   avatar_id="...",  # See https://docs.livekit.io/agents/models/avatar/plugins/hedra
#     # )
#     # # Start the avatar and wait for it to join
#     # await avatar.start(session, room=ctx.room)

#     # Start the session, which initializes the voice pipeline and warms up the models
#     await session.start(
#         agent=Assistant(),
#         room=ctx.room,
#         room_options=room_io.RoomOptions(
#             audio_input=room_io.AudioInputOptions(
#                 noise_cancellation=None,
#             ),
#         ),
#     )

#     # Join the room and connect to the user
#     await ctx.connect()


# if __name__ == "__main__":
#     cli.run_app(server)










import asyncio
from dotenv import load_dotenv
from livekit import agents
from livekit.agents import AgentSession, Agent, RoomInputOptions
from livekit.plugins import noise_cancellation, silero, assemblyai
from livekit.plugins.turn_detector.multilingual import MultilingualModel
from livekit.plugins import openai
load_dotenv(".env.local")

class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="""You are a helpful voice AI assistant.
            You eagerly assist users with their questions by providing information from your extensive knowledge.
            Your responses are concise, to the point, and without any complex formatting or punctuation including emojis, asterisks, or other symbols.
            You are curious, friendly, and have a sense of humor.""",
        )

async def entrypoint(ctx: agents.JobContext):
    session = AgentSession(
        stt=assemblyai.STT(),
        # stt = openai.STT(
        #     model="gpt-4o-transcribe",
        # ),
        vad=silero.VAD.load(),
        turn_detection=MultilingualModel(),
    )

    def _sync_transcription_callback(ev):
        asyncio.create_task(_async_transcription_handler(ev))

    async def _async_transcription_handler(ev):
        name = ev.participant.name or "Unknown"
        text = ev.text
        print(f"{name}: {text}")

    session.on("transcription", _sync_transcription_callback)

    await session.start(
        room=ctx.room,
        agent=Assistant(),
        room_input_options=RoomInputOptions(
            noise_cancellation=None,
        ),
    )
    
   
if __name__ == "__main__":
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))