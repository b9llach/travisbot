
## Features

- Conversational AI using Qwen2.5-14B-Instruct
- Music playback and queue management
- Math problem solving
- Practice question generation
- Internet search capabilities
- Course information management
- Voice channel support
- PDF processing

## Hardware Notes

This bot is configured to run on dual NVIDIA RTX 4090s using model parallelism. The model is loaded with `device_map="balanced"` to automatically distribute across available GPUs.

## Commands

- `/add_course_info` - Add course information
- `/remove_course_info` - Remove course information
- `/list_course_info` - List all course information
- `/reset_conversation` - Reset conversation history

Music commands are handled through natural language in mentions:
- Play music: `@bot play <song>`
- Queue management: `@bot queue`
- Volume control: `@bot volume <0-100>`
- Skip: `@bot skip`
- Pause/Resume: `@bot pause/resume`

### 1. Discord Bot Setup
1. Create a new application at [Discord Developer Portal](https://discord.com/developers/applications)
2. Create a bot under the application
3. Enable all Privileged Gateway Intents:
   - Presence Intent
   - Server Members Intent
   - Message Content Intent
4. Copy the bot token to your `.env` file
5. Invite the bot to your server using the OAuth2 URL generator
   - Select 'bot' and 'applications.commands' scopes
   - Select necessary permissions (Admin recommended for all features)

### 2. Run the Bot
Make sure your virtual environment is activated:

`python -m venv .venv` creates a new python virtual environment.
Windows:
`./.venv/scripts/activate`
Linux:
`. .venv/bin/activate`

Once activated, run the bot:
`python persona_bot.py`


