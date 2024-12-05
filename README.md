
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

