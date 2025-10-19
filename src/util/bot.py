from discord.ext import commands
import discord
import os
import asyncio
from queue import Queue
from rich.console import Console

console = Console()

description = "rarrhhhh"

intents = discord.Intents.default()
intents.members = True
intents.message_content = True

bot = commands.Bot(command_prefix="!", description=description, intents=intents)
_bot_is_ready = False

message_queue = Queue()


@bot.event
async def on_ready():
    global _bot_is_ready
    assert bot.user is not None

    console.print(f"bot activiated and online as {bot.user}")

    _bot_is_ready = True


def is_bot_ready():
    return _bot_is_ready


async def send_to_general(message):
    if not _bot_is_ready:
        print("Bot is not ready yet!")
        return

    for guild in bot.guilds:
        channel = discord.utils.get(guild.text_channels, name="general")
        if channel:
            await channel.send(message)
            return


async def messenger():
    while True:
        if not message_queue.empty():
            message = message_queue.get()
            await send_to_general(message)
        await asyncio.sleep(0.1)


async def start_bot():
    await bot.start(os.getenv("DISCORD_BOT_TOKEN"))


async def stop_bot():
    await bot.close()
