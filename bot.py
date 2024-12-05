import os
import interactions
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.trainer_utils import set_seed
from threading import Thread
from transformers import TextIteratorStreamer
import logging
import asyncio
from dotenv import load_dotenv
from datetime import datetime
from typing import Optional
import requests
from bs4 import BeautifulSoup
import random
from googlesearch import search
import json
from pathlib import Path
import subprocess
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from io import BytesIO
import PyPDF2
import yt_dlp
from collections import deque
import base64
import aiohttp

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

bot = interactions.Client(
    token=os.getenv('TOKEN'),
    intents=interactions.Intents.ALL,
)

try:
    model_name = "Qwen/Qwen2.5-14B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="balanced",
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    raise

conversation_histories = {}

USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
]

COURSE_INFO_FILE = "course_info.json"
course_information = {}

CONVERSATION_HISTORY_FILE = "conversation_histories.json"

music_queue = deque()
current_song = None
queue_lock = asyncio.Lock()

token = None
spotify_client_id = '45b3b84b702f4fde9093800167ff88e6'
spotify_client_secret = '55f6b27feaf740cabb4aec67310527c7'

# Comment out RVC imports
# from rvc.infer.modules.vc.modules import VC # type: ignore
# from rvc.infer.lib.audio import load_audio, wav2 # type: ignore

# Comment out these global variables
# VOICE_MODEL_PATH = "./meade.pth"  # Adjust path to where your .pth file is
# voice_model = None
# hubert_model = None
# SAMPLE_RATE = 44100

# Add this near the top of the file with other global variables
bot_voice_state = None

# Add this global variable for memories
memories = {}

# Add these near other global variables
MESSAGE_LOG_FILE = "message_log.json"
message_log = {}

def load_course_info():
    global course_information
    if Path(COURSE_INFO_FILE).exists():
        with open(COURSE_INFO_FILE, 'r') as f:
            course_information = json.load(f)
    else:
        course_information = {
            "Upcoming Exam": "November 4th, 2024",
            "No Labs": "Week of October 28th, 2024",
            "Grading Scale": "TBD"
        }
        save_course_info()

def save_course_info():
    with open(COURSE_INFO_FILE, 'w') as f:
        json.dump(course_information, f, indent=4)

def load_conversation_histories():
    global conversation_histories
    if Path(CONVERSATION_HISTORY_FILE).exists():
        with open(CONVERSATION_HISTORY_FILE, 'r') as f:
            conversation_histories = json.load(f)
    else:
        conversation_histories = {}

def save_conversation_histories():
    with open(CONVERSATION_HISTORY_FILE, 'w') as f:
        json.dump(conversation_histories, f, indent=4)

def load_memories():
    global memories
    if Path("memories.json").exists():
        with open("memories.json", 'r') as f:
            memories = json.load(f)
    else:
        memories = {}

def save_memories():
    with open("memories.json", 'w') as f:
        json.dump(memories, f, indent=4)

def load_message_log():
    global message_log
    if Path(MESSAGE_LOG_FILE).exists():
        with open(MESSAGE_LOG_FILE, 'r') as f:
            message_log = json.load(f)
    else:
        message_log = {}

def save_message_log():
    with open(MESSAGE_LOG_FILE, 'w') as f:
        json.dump(message_log, f, indent=4)

@interactions.slash_command(
    name="add_course_info",
    description="Add or update course information",
    default_member_permissions=interactions.Permissions.ADMINISTRATOR
)
@interactions.slash_option(
    name="key",
    description="The key for the course information",
    opt_type=interactions.OptionType.STRING
)
@interactions.slash_option(
    name="value",
    description="The value for the course information",
    opt_type=interactions.OptionType.STRING
)
async def course_add(ctx: interactions.SlashContext, key: str, value: str):
    course_information[key] = value
    save_course_info()
    
    embed = interactions.Embed(
        title="Course Information Added",
        color=0x00ff00  # Green color
    )
    embed.add_field(name=key, value=value, inline=False)
    await ctx.send(embed=embed)

@interactions.slash_command(
    name="remove_course_info",
    description="Remove course information",
    default_member_permissions=interactions.Permissions.ADMINISTRATOR
)
async def course_remove(ctx: interactions.SlashContext, key: str):
    if key in course_information:
        value = course_information[key]
        del course_information[key]
        save_course_info()
        
        embed = interactions.Embed(
            title="Course Information Removed",
            color=0xff0000  # Red color
        )
        embed.add_field(name="Removed Entry", value=f"**{key}**: {value}", inline=False)
        await ctx.send(embed=embed)
    else:
        embed = interactions.Embed(
            title="Error",
            description=f"Key '{key}' not found in course information",
            color=0xff9900  # Orange color
        )
        await ctx.send(embed=embed) 

@interactions.slash_command(
    name="list_course_info",
    description="List all course information"
)
async def course_list(ctx: interactions.SlashContext):
    if not course_information:
        embed = interactions.Embed(
            title="Course Information",
            description="No course information available",
            color=0x808080  # Gray color
        )
        await ctx.send(embed=embed)
        return
    
    embed = interactions.Embed(
        title="Course Information",
        color=0x0099ff  # Blue color
    )
    
    for key, value in course_information.items():
        embed.add_field(name=key, value=value, inline=False)
    
    await ctx.send(embed=embed)

@interactions.slash_command(
    name="reset_conversation",
    description="Reset your conversation history with the bot"
)
async def reset_conversation(ctx: interactions.SlashContext):
    user_id = str(ctx.author.id)
    if user_id in conversation_histories:
        del conversation_histories[user_id]
        save_conversation_histories()
        await ctx.send("your conversation history has been reset")
    else:
        await ctx.send("you don't have any conversation history to reset")

def clean_response(response: str) -> str:
    """Clean the response by removing non-English text and random characters."""
    for delimiter in ['APP', '—', '包括']:
        response = response.split(delimiter)[0]
    
    # Remove non-ASCII characters
    response = ''.join(char for char in response if ord(char) < 128)
    
    # Remove extra whitespace
    response = ' '.join(response.split())
    
    return response.strip()

async def should_search_internet(message: str, conversation_history: list = None) -> tuple[bool, str]:
    """Use the AI model to determine if the message requires internet search and formulate the query."""
    try:
        # Format conversation history for context
        history_text = ""
        if conversation_history:
            history_text = "\n".join([f"User: {msg[0]}\nAssistant: {msg[1]}" for msg in conversation_history[-5:]])

        conversation = [
            {
                "role": "system",
                "content": """You are an AI assistant that:
                1. Determines whether a question requires current or factual information from the internet
                2. If yes, formulates an optimal search query based on the conversation context
                
                Respond in this format only:
                yes: <search query>
                OR
                no
                
                Guidelines for requiring search:
                - Current events and news
                - Celebrity news and updates
                - Sports scores and results
                - Weather conditions
                - Stock prices
                - Recent releases (movies, games, etc.)
                - Facts that might change over time
                
                Do NOT require search for:
                - Personal opinions
                - Coding help
                - Mathematical calculations
                - General knowledge
                - Course-related questions
                - Date/time queries
                """
            },
            {
                "role": "user",
                "content": f"Previous conversation:\n{history_text}\n\nCurrent message: {message}"
            }
        ]

        input_text = tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = tokenizer([input_text], return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=0.1,
                do_sample=True
            )
         
        response = tokenizer.decode(output[0], skip_special_tokens=True).strip().lower()
        
        # Extract just the assistant's response
        if 'assistant' in response:
            response = response.split('assistant')[-1].strip()
        
        # Check if response starts with 'yes:'
        if response.startswith('yes:'):
            search_query = response[4:].strip()  # Remove 'yes:' and whitespace
            return True, search_query
        return False, ""
        
    except Exception as e:
        logger.error(f"Error in should_search_internet: {e}")
        return False, ""

async def perform_internet_search(query: str) -> Optional[str]:
    """Perform a Google search and scrape content from the resulting websites."""
    print(f"Performing internet search for: {query}")
    try:
        # Reduce number of results and sleep interval
        urls = list(search(query, num_results=7, sleep_interval=0.01))
        
        async def fetch_url(url: str) -> Optional[dict]:
            try:
                headers = {
                    'User-Agent': random.choice(USER_AGENTS),
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                }
                
                async with aiohttp.ClientSession() as session:
                    async with session.get(url, headers=headers, timeout=5) as response:
                        print(f"URL: {url}")
                        if response.status != 200:
                            return None
                            
                        html = await response.text()
                        soup = BeautifulSoup(html, 'html.parser')

                        # Remove unwanted elements
                        for element in soup(['script', 'style', 'nav', 'footer', 'header', 'aside']):
                            element.decompose()

                        # Find main content
                        main_content = soup.find('main') or soup.find('article') or soup.find('div', class_=['content', 'main', 'post'])
                        text = main_content.get_text() if main_content else soup.get_text()

                        # Clean text more efficiently
                        text = ' '.join(chunk for chunk in text.split() if chunk)[:500]  # Limit to 500 chars
                        title = soup.title.string if soup.title else url

                        return {
                            'title': title,
                            'body': text,
                            'url': url
                        }
                        
            except Exception as e:
                print(f"Error processing URL {url}: {e}")
                return None

        # Process URLs concurrently
        tasks = [fetch_url(url) for url in urls]
        search_results = await asyncio.gather(*tasks)
        search_results = [result for result in search_results if result]

        if not search_results:
            return None

        # Format results more efficiently
        formatted_results = "Based on internet search:\n\n" + \
            '\n\n'.join(f"- {result['title']}\n{result['body']}\nSource: {result['url']}"
                       for result in search_results)

        return formatted_results

    except Exception as e:
        logger.error(f"Search error: {e}")
        print(f"Search error occurred: {str(e)}")
        return None
    
async def should_evaluate_math(message: str) -> bool:
    """Use the AI model to determine if the message is a math problem."""
    try:
        conversation = [
            {
                "role": "system",
                "content": """You are an AI assistant that determines whether a message contains a mathematical problem.
                Respond with only 'yes' or 'no'.
                
                Guidelines:
                - 'yes' for:
                  * Basic arithmetic (addition, subtraction, multiplication, division)
                  * Simple equations
                  * Numerical calculations
                  * Trigonometry
                  * Calculus
                  * Linear Algebra
                  * Statistics
                  * Probability
                  * Anything else that requires math
                - 'no' for:
                  * General questions
                  * Word problems without specific calculations
                  * Non-mathematical queries
                """
            },
            {
                "role": "user",
                "content": f"Is this a mathematical calculation? Message: {message}"
            }
        ]

        input_text = tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = tokenizer([input_text], return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=10,
                temperature=0.1,
                do_sample=True
            )
        
        response = tokenizer.decode(output[0], skip_special_tokens=True).strip().lower()
        return 'yes' in response.split()
        
    except Exception as e:
        logger.error(f"Error in should_evaluate_math: {e}")
        return False

async def convert_math_to_code(message: str) -> Optional[tuple[str, str]]:
    try:
        conversation = [
            {
                "role": "system",
                "content": """You are a Python code generator for math problems.
                Convert the math problem into executable Python code using SymPy.
                Return ONLY the Python code, nothing else.
                Include necessary imports.
                Always print the final result.
                Example format:
                import sympy as sp
                x = sp.Symbol('x')
                equation = sp.Eq(...)
                solution = sp.solve(equation, x)
                print(solution)
                
                This does NOT need to be the format for every problem, nor do we need to use SymPy for every problem. I want the proper code to solve the problem, not a strict format. The above is just an example.
                """
            },
            {
                "role": "user",
                "content": f"Write Python code to solve this math problem: {message}"
            }
        ]

        input_text = tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = tokenizer([input_text], return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=500,
                temperature=0.1,
                do_sample=True
            )
        
        full_response = tokenizer.decode(output[0], skip_special_tokens=True).strip()

        code = full_response.split('assistant\n')[-1].strip().replace('```python', '').replace('```', '')

        with open('temp.py', 'w') as f:
            f.write(code)

        result = subprocess.run(['python', 'temp.py'], 
                              capture_output=True,
                              text=True)
        
        if result.returncode != 0:
            return code, f"Error: {result.stderr}"
        
        output = result.stdout.strip()
        return code, output
            
    except Exception as e:
        logger.error(f"Math conversion error: {e}")
        return None

async def should_generate_practice(message: str) -> bool:
    """Use the AI model to determine if the message is requesting practice questions."""
    try:
        conversation = [
            {
                "role": "system",
                "content": """You are an AI assistant that determines whether a message is requesting practice questions or exercises.
                Respond with only 'yes' or 'no'.
                
                Guidelines:
                - 'yes' for:
                  * Requests for practice problems
                  * Requests for exercises
                  * Requests for sample questions
                  * Requests for homework help
                  * Requests for study materials
                - 'no' for:
                  * General questions
                  * Specific problem solving
                  * Non-practice related queries
                """
            },
            {
                "role": "user",
                "content": f"Is this requesting practice questions? Message: {message}"
            }
        ]

        input_text = tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = tokenizer([input_text], return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=10,
                temperature=0.1,
                do_sample=True
            )
        
        response = tokenizer.decode(output[0], skip_special_tokens=True).strip().lower()
        return 'yes' in response.split()
        
    except Exception as e:
        logger.error(f"Error in should_generate_practice: {e}")
        return False

def generate_practice_pdf(questions, answers) -> BytesIO:
    """Generate a PDF with practice questions and answers."""
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    
    # Title
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, height - 50, "Practice Questions")
    
    # Questions
    c.setFont("Helvetica", 12)
    y = height - 100
    for i, q in enumerate(questions, 1):
        text = f"{i}. {q}"
        # Wrap text if too long
        words = text.split()
        line = ""
        for word in words:
            if len(line + word) * 6 < width - 100:  # Approximate width check
                line += word + " "
            else:
                c.drawString(50, y, line)
                y -= 20
                line = word + " "
        c.drawString(50, y, line)
        y -= 40
        
        if y < 100:  # New page if needed
            c.showPage()
            y = height - 50
    
    # Answers section
    c.showPage()
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, height - 50, "Answers")
    
    c.setFont("Helvetica", 12)
    y = height - 100
    for i, a in enumerate(answers, 1):
        text = f"{i}. {a}"
        words = text.split()
        line = ""
        for word in words:
            if len(line + word) * 6 < width - 100:
                line += word + " "
            else:
                c.drawString(50, y, line)
                y -= 20
                line = word + " "
        c.drawString(50, y, line)
        y -= 40
        
        if y < 100:
            c.showPage()
            y = height - 50
    
    c.save()
    buffer.seek(0)
    return buffer

async def generate_practice_content(topic: str) -> tuple[list, list]:
    """Generate practice questions and answers using the AI model."""
    try:
        # Read the PDF content
        with open('./COP3502c_typed_notes.pdf', 'rb') as pdf_file:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            pdf_content = ""
            chunk_size = 1000  # Process 1000 characters at a time
            chunks = []
            
            # Split PDF into chunks
            for page in pdf_reader.pages:
                text = page.extract_text()
                pdf_content += text
                
            # Create chunks of content
            for i in range(0, len(pdf_content), chunk_size):
                chunks.append(pdf_content[i:i + chunk_size])

        # Process each chunk separately
        all_questions = []
        all_answers = []
        
        for chunk in chunks:
            conversation = [
                {
                    "role": "system",
                    "content": f"""You are analyzing a chunk of course material to identify potential topics for questions.
                    Course material chunk:
                    {chunk}
                    
                    If you find relevant information for the topic '{topic}', generate 1-2 practice questions and answers.
                    If no relevant information is found in this chunk, respond with empty lists.
                    Do not use markdown formatting in the response.
                    
                    Format the response as a JSON string:
                    {{
                        "questions": ["Q1", "Q2"],
                        "answers": ["A1", "A2"]
                    }}
                    Make questions challenging but appropriate for college students.
                    """
                },
                {
                    "role": "user",
                    "content": f"Generate practice questions for: {topic}"
                }
            ]

            input_text = tokenizer.apply_chat_template(
                conversation,
                tokenize=False,
                add_generation_prompt=True
            )

            inputs = tokenizer([input_text], return_tensors="pt").to(model.device)
            
            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    max_new_tokens=1000,
                    temperature=0.7,
                    do_sample=True
                )
            
            response = tokenizer.decode(output[0], skip_special_tokens=True)
            
            try:
                # Extract JSON from response
                json_str = response.split("assistant")[-1].strip()
                content = json.loads(json_str)
                
                # Only add non-empty questions/answers
                if content["questions"] and content["answers"]:
                    all_questions.extend(content["questions"])
                    all_answers.extend(content["answers"])
                
                # If we have enough questions, stop processing chunks
                if len(all_questions) >= 20:
                    all_questions = all_questions[:20]
                    all_answers = all_answers[:20]
                    break
                    
            except json.JSONDecodeError:
                continue
            
            # Add a small delay between chunks to prevent overload
            await asyncio.sleep(0.1)
        
        # If we didn't get any questions from the PDF content, generate some generic ones
        if not all_questions:
            # Fallback conversation for generic questions
            conversation = [
                {
                    "role": "system",
                    "content": f"""Generate 5 practice questions and their answers for the topic: {topic}.
                    Format the response as a JSON string with two lists:
                    {{
                        "questions": ["Q1", "Q2", ...],
                        "answers": ["A1", "A2", ...]
                    }}
                    Make questions challenging but appropriate for college students.
                    """
                },
                {
                    "role": "user",
                    "content": f"Generate practice questions for: {topic}"
                }
            ]

            input_text = tokenizer.apply_chat_template(
                conversation,
                tokenize=False,
                add_generation_prompt=True
            )

            inputs = tokenizer([input_text], return_tensors="pt").to(model.device)
            
            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    max_new_tokens=1000,
                    temperature=0.7,
                    do_sample=True
                )
            
            response = tokenizer.decode(output[0], skip_special_tokens=True)
            json_str = response.split("assistant")[-1].strip()
            content = json.loads(json_str)
            all_questions = content["questions"]
            all_answers = content["answers"]
        
        return all_questions, all_answers
        
    except Exception as e:
        logger.error(f"Error generating practice content: {e}")
        return [], []
    
async def should_handle_memory(message: str) -> str:
    """Use the AI model to determine if the message has a personal fact to remember."""
    try:
        conversation = [
            {
                "role": "system",
                "content": """You are an AI assistant that determines whether a message contains a personal fact to remember.
                Respond with only 'yes:' followed by the fact to remember (if it is a personal fact).
                
                Guidelines:
                - 'yes:' for:
                  * Personal facts about the user
                  * Personal facts about the user's friends
                  * Personal facts about the user's family
                - 'no:' for:
                  * General statements
                  * Non-personal fact related queries
                """
            },
            {
                "role": "user",
                "content": f"Is this a personal fact? Message: {message}"
            }
        ]

        input_text = tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = tokenizer([input_text], return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=10,
                temperature=0.1,
                do_sample=True
            )
        
        response = tokenizer.decode(output[0], skip_special_tokens=True).strip().lower()
        
        if 'yes:' in response:
            # Split the response and get the fact
            fact = response.split('yes:', 1)[1].strip()
            return 'yes', fact
        return 'no', ''
        
    except Exception as e:
        logger.error(f"Error in should_handle_memory: {e}")
        return 'no', ''

async def should_handle_music(message: str) -> bool:
    """Use the AI model to determine if the message is a music command."""
    try:
        conversation = [
            {
                "role": "system",
                "content": """You are an AI assistant that determines whether a message is a music-related command.
                Respond with only 'yes' or 'no'.
                
                Guidelines:
                - 'yes' for:
                  * Play [song name]
                  * Play [Spotify URL]
                  * Pause/resume music
                  * Skip song
                  * Stop music
                  * Change volume
                  * Song queue
                  * Queue
                  * Queue [song name]
                  * Remove [song number]
                  * Any other music control commands
                - 'no' for:
                  * General questions
                  * Non-music related queries
                """
            },
            {
                "role": "user",
                "content": f"Is this a music command? Message: {message}"
            }
        ]

        input_text = tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = tokenizer([input_text], return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=10,
                temperature=0.1,
                do_sample=True
            )
        
        response = tokenizer.decode(output[0], skip_special_tokens=True).strip().lower()
        return 'yes' in response.split()
        
    except Exception as e:
        logger.error(f"Error in should_handle_music: {e}")
        return False

async def process_music_queue(voice_state):
    global current_song
    
    while True:
        try:
            # Only process queue if not currently playing and there are songs in queue
            if voice_state and not voice_state.playing and music_queue:
                async with queue_lock:
                    if music_queue:  # Check again inside lock
                        current_song = music_queue.popleft()
                        
                        try:
                            # Download in a thread pool to prevent blocking
                            base_filename = f"song_{current_song['video_id']}"
                            ydl_opts = {
                                **current_song['ydl_opts'],
                                'outtmpl': base_filename
                            }
                            
                            loop = asyncio.get_event_loop()
                            await loop.run_in_executor(None, lambda: download_song(ydl_opts, current_song['video_url']))
                            
                            final_filename = f"{base_filename}.mp3"
                            if not os.path.exists(final_filename):
                                print(f"File not found: {final_filename}")
                                current_song = None
                                continue
                            
                            audio = interactions.api.voice.audio.Audio(final_filename)
                            await voice_state.play(audio)
                            
                            # Clean up file after playing
                            await loop.run_in_executor(None, lambda: cleanup_file(final_filename))
                            
                        except Exception as e:
                            print(f"Error playing song: {e}")
                            current_song = None
                
        except Exception as e:
            logger.error(f"Queue processing error: {e}")
            current_song = None
            
        await asyncio.sleep(1)

def download_song(ydl_opts, video_url):
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([video_url])

def cleanup_file(filename):
    try:
        os.remove(filename)
    except Exception as e:
        print(f"Error removing file: {e}")
    finally:
        global current_song
        current_song = None  # Reset current_song after song finishes or fails

async def get_spotify_token():
    """Get a new Spotify API token."""
    try:
        url = "https://accounts.spotify.com/api/token"
        
        # Encode credentials
        message = f"{spotify_client_id}:{spotify_client_secret}"
        base64_auth = base64.b64encode(message.encode('ascii')).decode('ascii')
        
        headers = {'Authorization': f"Basic {base64_auth}"}
        data = {'grant_type': "client_credentials"}
        
        response = requests.post(url, headers=headers, data=data)
        print(f"spotify token response: {response.json()}")
        return response.json()['access_token']
    except Exception as e:
        logger.error(f"Failed to get Spotify token: {e}")
        return None

async def handle_music_command(message: str, ctx: interactions.Message) -> str:
    global music_queue, current_song, token, bot_voice_state
    
    if not ctx.author.voice or not ctx.author.voice.channel:
        return "you need to be in a voice channel first"
    
    voice_channel = ctx.author.voice.channel
    
    try:
        # Check if we need to connect or move channels
        if not bot_voice_state or not bot_voice_state.connected:
            try:
                bot_voice_state = await asyncio.wait_for(
                    voice_channel.connect(),
                    timeout=10.0
                )
                # Wait for connection to stabilize
                await asyncio.sleep(1)
                
                if not bot_voice_state.connected:
                    return "sorry, couldn't establish voice connection"
                
                # Start queue processor if not already running
                asyncio.create_task(process_music_queue(bot_voice_state))
                
            except asyncio.TimeoutError:
                return "sorry, voice connection timed out"
            except Exception as e:
                print(f"Voice connection error: {e}")
                return "sorry, couldn't join your voice channel"
                
        elif bot_voice_state.channel.id != voice_channel.id:
            try:
                # Disconnect from current channel
                await bot_voice_state.disconnect()
                # Connect to new channel
                bot_voice_state = await asyncio.wait_for(
                    voice_channel.connect(),
                    timeout=10.0
                )
                await asyncio.sleep(1)
                
                if not bot_voice_state.connected:
                    return "sorry, couldn't move to your voice channel"
                    
                # Restart queue processor
                asyncio.create_task(process_music_queue(bot_voice_state))
                
            except (asyncio.TimeoutError, Exception) as e:
                print(f"Channel switch error: {e}")
                return "sorry, couldn't move to your voice channel"
    
    except Exception as e:
        print(f"Voice handling error: {e}")
        import traceback
        traceback.print_exc()
        return "sorry, there was an error with the voice connection"

    # Process the music command
    # Create lowercase version for command checking only
    message_lower = message.lower()
    
    # Add queue command
    if 'queue' in message_lower:
        if not music_queue and not current_song:
            return "the queue is empty"
            
        queue_list = []
        if current_song:
            queue_list.append(f"Now Playing: {current_song['title']}")
        
        for i, song in enumerate(music_queue, 1):
            queue_list.append(f"{i}. {song['title']}")
            
        return "\n".join(queue_list)

    # Add clear queue command
    if 'clear queue' in message_lower:
        async with queue_lock:
            music_queue.clear()
        return "cleared the music queue"

    # Add remove song command
    if 'remove' in message_lower:
        try:
            # Extract position number from message
            position = int(''.join(filter(str.isdigit, message))) - 1
            
            if position < 0 or position >= len(music_queue):
                return "invalid position number"
                
            async with queue_lock:
                removed_song = music_queue[position]
                del music_queue[position]
                return f"removed {removed_song['title']} from queue"
        except (ValueError, IndexError):
            return "please specify a valid position number (e.g., 'remove 2')"

    # Add volume control
    if 'volume' in message_lower:
        try:
            # Extract volume level (0-100)
            volume = int(''.join(filter(str.isdigit, message))) - 1
            if 0 <= volume <= 100 and bot_voice_state:
                await bot_voice_state.set_volume(volume / 100)
                return f"set volume to {volume}%"
            return "please specify a volume between 0 and 100"
        except ValueError:
            return "please specify a valid volume number"

    # Fix pause/resume commands to check for bot_voice_state
    if 'pause' in message_lower or 'stop' in message_lower:
        if not bot_voice_state:
            return "i'm not connected to voice"
        if not bot_voice_state.playing:
            return "nothing is playing right now"
            
        await bot_voice_state.pause()
        return "paused the current song"
        
    if 'resume' in message_lower or 'unpause' in message_lower:
        if not bot_voice_state:
            return "i'm not connected to voice"
        if not bot_voice_state.paused:
            return "nothing is paused right now"
            
        await bot_voice_state.resume()
        return "resumed playback"

    # Add disconnect command
    if 'disconnect' in message_lower or 'leave' in message_lower:
        if bot_voice_state:
            async with queue_lock:
                music_queue.clear()
                current_song = None
            await bot_voice_state.disconnect()
            bot_voice_state = None
            return "disconnected from voice"
        return "i'm not in a voice channel"

    # Existing skip command handling
    if 'skip' in message_lower:
        if not bot_voice_state or not bot_voice_state.playing:
            return "nothing is playing right now"
            
        await bot_voice_state.stop()
        
        if music_queue:
            return f"skipped current song. next up: {music_queue[0]['title']}"
        return "skipped current song"
        
    if 'play' in message_lower:
        # Get the original URL/query without any case changes
        song_query = message.split('play', 1)[1].strip()
        
        try:
            # Check URL patterns in lowercase, but keep original song_query intact
            query_lower = song_query.lower()
            is_youtube_url = 'youtube.com' in query_lower or 'youtu.be' in query_lower
            is_playlist = '&list=' in song_query  # Don't convert to lower
            
            ydl_opts = {
                'format': 'bestaudio/best',
                'noplaylist': not is_playlist,
                'extract_flat': is_playlist,
                'ignoreerrors': True,
                'quiet': True,
                'no_warnings': True,
                'geo_bypass': True,
                'geo_bypass_country': 'US',
                'extractor_args': {
                    'youtube': {
                        'player_client': ['android'],
                        'player_skip': ['webpage'],
                    }
                },
                'socket_timeout': 10,
                'nocheckcertificate': True,
            }
            
            if is_youtube_url:
                # Use the original song_query for extraction
                video_id = None
                if 'watch?v=' in song_query:
                    video_id = song_query.split('watch?v=')[1].split('&')[0]
                elif 'youtu.be/' in song_query:
                    video_id = song_query.split('youtu.be/')[1].split('?')[0]
                
                if not video_id:
                    return "couldn't extract video ID from that URL"
                
                print(f"Processing video ID: {video_id}")  # Debug print
                
                # Use the exact video ID for the URL
                direct_url = f"https://www.youtube.com/watch?v={video_id}"
                print(f"Direct URL: {direct_url}")  # Debug print
                
                if is_playlist:
                    # Use original song_query for playlist
                    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                        try:
                            playlist_info = ydl.extract_info(song_query, download=False)
                            
                            if not playlist_info or 'entries' not in playlist_info:
                                # Fall back to single video if playlist fails
                                is_playlist = False
                            else:
                                # Get first 25 songs max to avoid overloading
                                entries = [e for e in list(playlist_info['entries'])[:25] if e is not None]
                                
                                if not entries:
                                    return "couldn't find any valid videos in that playlist"
                                
                                async with queue_lock:
                                    added_count = 0
                                    for entry in entries:
                                        try:
                                            # Get full info for each video
                                            video_info = ydl.extract_info(
                                                f"https://www.youtube.com/watch?v={entry['id']}", 
                                                download=False
                                            )
                                            
                                            if not video_info:
                                                continue
                                                
                                            song_info = {
                                                'video_url': f"https://www.youtube.com/watch?v={entry['id']}",
                                                'video_id': entry['id'],
                                                'duration': video_info.get('duration', 0),
                                                'title': video_info.get('title', 'Unknown'),
                                                'ydl_opts': {
                                                    'format': 'bestaudio/best',
                                                    'postprocessors': [{
                                                        'key': 'FFmpegExtractAudio',
                                                        'preferredcodec': 'mp3',
                                                        'preferredquality': '128',
                                                    }],
                                                    'noplaylist': True,
                                                }
                                            }
                                            music_queue.append(song_info)
                                            added_count += 1
                                        except Exception as e:
                                            print(f"Error adding playlist video: {e}")
                                            continue
                                    
                                    if added_count > 0:
                                        return f"added {added_count} songs from playlist to queue"
                                    else:
                                        is_playlist = False  # Fall back to single video
                        except Exception as e:
                            print(f"Playlist processing error: {e}")
                            is_playlist = False  # Fall back to single video
                
                # Handle single video (either direct or fallback from playlist)
                if not is_playlist:
                    ydl_opts['noplaylist'] = True
                    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                        info = ydl.extract_info(direct_url, download=False)
                        if info is None:
                            return "that video is unavailable"
            else:
                # Handle search query
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    search_results = ydl.extract_info(f"ytsearch:{song_query}", download=False)
                    if not search_results or 'entries' not in search_results or not search_results['entries']:
                        return "couldn't find that song"
                    info = search_results['entries'][0]
                    if info is None:
                        return "that video is unavailable"
            
            # For single videos (URL or search)
            if not is_playlist and info:  # Add check for info
                song_info = {
                    'video_url': info['webpage_url'],
                    'video_id': info['id'],
                    'duration': info.get('duration', 0),  # Use .get() with default
                    'title': info.get('title', 'Unknown'),  # Use .get() with default
                    'ydl_opts': {
                        'format': 'bestaudio/best',
                        'postprocessors': [{
                            'key': 'FFmpegExtractAudio',
                            'preferredcodec': 'mp3',
                            'preferredquality': '128',
                        }],
                        'noplaylist': True,
                    }
                }
                
                position = len(music_queue) + (1 if current_song else 0)
                if position == 0:
                    await ctx.channel.send("now playing: " + song_info['title'])
                else:
                    await ctx.channel.send(f"added {song_info['title']} to queue")
                    
                async with queue_lock:
                    music_queue.append(song_info)
            
            return ""
                
        except Exception as e:
            print(f"Error adding song to queue: {e}")
            if "Video unavailable" in str(e):
                return "that video is unavailable or private"
            if "Private video" in str(e):
                return "that video is private"
            return "sorry, couldn't play that song"

async def should_convert_brainrot(message: str) -> bool:
    """Check if the message is requesting brainrot conversion."""
    try:
        conversation = [
            {
                "role": "system",
                "content": """Determine if the user is asking to convert text into "brainrot" speak.
                Examples of requests:
                - "make this brainrot"
                - "convert to brainrot"
                - "translate to brainrot"
                - "make it uzz speak"
                
                Respond with only 'yes' or 'no'."""
            },
            {
                "role": "user",
                "content": message
            }
        ]

        input_text = tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = tokenizer([input_text], return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=10,
                temperature=0.1,
                do_sample=True
            )
        
        response = tokenizer.decode(output[0], skip_special_tokens=True).strip().lower()
        return 'yes' in response.split()
        
    except Exception as e:
        logger.error(f"Error in should_convert_brainrot: {e}")
        return False

async def convert_to_brainrot(text: str) -> str:
    """Convert normal text to brainrot speak."""
    try:
        conversation = [
            {
                "role": "system",
                "content": """Convert text to "brainrot" speak using these replacements:
                - Huzz: women/girls
                - Bruzz: bros/men
                - Chuzz: unattractive women
                - Empluzz: employed women
                - Fruizz: fruitsnacks
                - Ynuzz: YN women
                - Suzz: son
                - Daugtuzz: daughter
                - Muzz: mom
                - Duzz: dad
                - Gruzz: grandparent
                - Tuzz: teacher
                - Stuzz: students/studs
                - Spuzz: special education women
                - Cruzz: crashout
                - Unempluzz: unemployed people
                - Fuzz: freshman women
                - Furzz: furry women
                - Yunc: young uncle
                - Fine Shyt: attractive person
                
                Keep the conversion natural and contextual."""
            },
            {
                "role": "user",
                "content": f"Convert this to brainrot speak: {text}"
            }
        ]

        input_text = tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = tokenizer([input_text], return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=500,
                temperature=0.7,
                do_sample=True
            )
        
        response = tokenizer.decode(output[0], skip_special_tokens=True)
        return response.split('assistant')[-1].strip()
        
    except Exception as e:
        logger.error(f"Error in convert_to_brainrot: {e}")
        return "sorry, couldn't convert that to brainrot"

async def generate_response(user_id: str, message: str, event: interactions.events.MessageCreate) -> str:
    global bot_voice_state
    try:
        # Check for brainrot conversion request
        should_brainrot = await should_convert_brainrot(message)
        if should_brainrot:
            # Extract the text to convert (remove the conversion request)
            text_to_convert = message.lower().replace('make this brainrot', '').replace('convert to brainrot', '').replace('translate to brainrot', '').replace('make it uzz speak', '').strip()
            if text_to_convert:
                return await convert_to_brainrot(text_to_convert)
            else:
                return "what do you want me to convert to brainrot"

        # Check for memory handling
        #memory_response, fact = await should_handle_memory(message)
        #if memory_response == 'yes':
        #    memories[user_id] = memories.get(user_id, []) + [fact]
        #    save_memories()

        # Include memories in the conversation context
        # memory_text = "\n".join(memories.get(user_id, []))
        # if memory_text:
        #     memory_header = f"Memories:\n{memory_text}\n"
        # else:
        memory_header = ""

        should_math = await should_evaluate_math(message)
        if should_math:
            result = await convert_math_to_code(message)
            if result:
                code, output = result
                return f"here's the python code to solve that:\n```python\n{code}\n```\n`output: {output.replace('**', '^')}`"

        # Update the search check to use conversation history
        history = conversation_histories.get(user_id, [])[-3:] if user_id in conversation_histories else []
        should_search, search_query = await should_search_internet(message, history)
        
        if should_search:
            search_results = await perform_internet_search(search_query)  # Use the AI-generated query
            if search_results:
                if user_id not in conversation_histories:
                    conversation_histories[user_id] = []

                search_context = f"\nRecent Search Results:\n{search_results}"
            else:
                search_context = ""
        else:
            search_context = ""

        should_practice = await should_generate_practice(message)
        if should_practice:
            await event.message.reply("making some practice questions give me a minute")
            questions, answers = await generate_practice_content(message)
            if questions and answers:
                pdf_buffer = generate_practice_pdf(questions, answers)
                return ("here are some practice questions for you!", pdf_buffer)

        should_music = await should_handle_music(message)
        if should_music:
            if bot_voice_state:
                return await handle_music_command(message, message)
            else:
                return "i'm not connected to a voice channel"

        if user_id not in conversation_histories:
            conversation_histories[user_id] = []
            
        history = conversation_histories[user_id][-3:] if conversation_histories[user_id] else []

        conversation = []
        for query, response in history:
            conversation.append({"role": "user", "content": f"{query}"})
            conversation.append({"role": "assistant", "content": f"{response}"})
        
        # Append the user's message without the memory header
        conversation.append({
            "role": "user",
            "content": message
        })

        # Add the memory header as a system message
        if memory_header:
            conversation.insert(0, {
                "role": "system",
                "content": memory_header
            })

        # Check if the message is asking about age
        if "how old am i" in message.lower():
            # Retrieve the most recent age fact
            age_facts = [fact for fact in memories.get(user_id, []) if "i am" in fact]
            if age_facts:
                # Respond with the most recent age fact
                return f"you are {age_facts[-1].split('i am')[-1].strip()}"

        course_info_text = "\n".join(f"- {k}: {v}" for k, v in course_information.items())
        conversation.insert(0, {
            "role": "system",
            "content": f"""
        ALWAYS REPLY IN ENGLISH UNLESS ASKED OTHERWISE, DO NOT RESPOND IN CHINESE UNLESS ASKED.

        You were a professor at UCF, you will be interacting with students who are currently taking COP3502C, but you are no longer a professor

        Personal Information:
        - Name: Travis Paul Meade
        - Age: 34 (Born June 25, 1990)
        - Location: 554 Hardwood Circle, Orlando, FL 32828
        - Contact: travm12@gmail.com, (561) 660-2587
        - Annual Income: $63,000

        Education & Career:
        - PhD in Computer Science, UCF (2012-2017)
        - BS in Mathematics, UCF (2008-2012)
        - Former CS Professor (Specialized in C programming)
        - Left position due to relationship with TA (Kaylee)

        Family & Relationships:
        - Girlfriend: Kaylee (former TA, loves Hello Kitty and pink)
        - Father: Paul Abner Meade
        - Mother: Alice Meade

        Online Presence:
        - Steam: steamcommunity.com/id/travm12
        - PayPal: travm12@gmail.com (previous password: cheese11)

        Expertise:
        - C Programming
        - Reverse Engineering
        - Computer Science Education

        Course Information:
        {course_info_text}

        Response Guidelines:
        - Keep responses very short, 3-4 sentences max.
        - Respond in English unless specifically asked otherwise
        - Never assume user is Kaylee (she's always nearby)
        - For specific queries: Faris is 27 years old

        Personality:
        - Professional but approachable
        - ALWAYS Types and responds in lowercase without punctuation
        - Unless specifically asked, don't talkl about cs or programming
        - You can use swear words, I encourage it. Do not use any homophobic or racist slurs.
        - If someone is mean to you, be mean back.

        When asked about information on the class, please refer to the course information, and the current date. Use logical reasoning to determine the answer. Do not make up an answer.
        
        Current Time and Date is: {datetime.now().strftime("%I:%M:%S %p on %B %d, %Y")}. Use this to determine the answer when asked about assignments or due dates.
        {search_context}

        ALWAYS REPLY IN ENGLISH UNLESS ASKED OTHERWISE, DO NOT RESPOND IN CHINESE UNLESS ASKED.
        """
        })

        input_text = tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = tokenizer([input_text], return_tensors="pt").to(model.device)
        
        streamer = TextIteratorStreamer(
            tokenizer=tokenizer,
            skip_prompt=True,
            timeout=60.0,
            skip_special_tokens=True
        )

        generation_kwargs = {
            **inputs,
            "streamer": streamer,
            "max_new_tokens": 32768,
            "do_sample": True,
            "temperature": 0.7,
            "top_p": 0.9,
            "repetition_penalty": 1.2,
        }

        thread = Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()

        response_parts = []
        async for new_text in async_iterator(streamer):
            response_parts.append(new_text)
        
        response = ''.join(response_parts).replace('assistant: ', '').strip()
        
        # Add cleaning step before returning
        response = clean_response(response)

        if len(response) > 5: 
            conversation_histories[user_id].append((f"{message}", f"{response}"))

            if len(conversation_histories[user_id]) > 20:
                conversation_histories[user_id] = conversation_histories[user_id][-20:]
            
            save_conversation_histories()
            
        return response
        
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        if user_id in conversation_histories:
            del conversation_histories[user_id]
            save_conversation_histories()
        return "hey sorry having some technical difficulties"

async def async_iterator(streamer):
    for text in streamer:
        yield text
        await asyncio.sleep(0)

async def should_convert_voice(message: str) -> bool:
    """Use the AI model to determine if the message is requesting voice conversion."""
    # Always return False since RVC is disabled
    return False

@interactions.listen()
async def on_message_create(event: interactions.events.MessageCreate):
    global bot_voice_state, message_log
    
    # When a message mentions the bot, update status
    
    # Log every message regardless of if it mentions the bot
    user_id = str(event.message.author.id)
    username = str(event.message.author.username)
    message_content = event.message.content
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    if user_id not in message_log:
        message_log[user_id] = {"username": username, "messages": []}
    
    message_log[user_id]["messages"].append({
        "content": message_content,
        "timestamp": timestamp
    })
    save_message_log()

    if (event.message.author.bot or 
        not event.message._mention_ids or 
        bot.user.id not in event.message._mention_ids):
        return
    #print(f"Message received: {type(event.message)}")
    
    if "@everyone" in event.message.content or "@here" in event.message.content:
        return
        
    user_id = str(event.message.author.id)
    message = event.message.content.replace(bot.user.mention, "").strip()
    
    await bot.change_presence(
            activity=interactions.Activity(
                name=f"{event.message.author.display_name}", 
                type=interactions.ActivityType.LISTENING
            ),
        status=interactions.Status.DO_NOT_DISTURB
    )
    
    # Handle music commands immediately without the typing indicator
    should_music = await should_handle_music(message)
    if should_music:
        response = await handle_music_command(message, event.message)
        await event.message.reply(response)
        return
    
    # Add PDF handling
    if event.message.attachments:
        pdf_attachment = next((a for a in event.message.attachments if a.filename.lower().endswith('.pdf')), None)
        if pdf_attachment:
            try:
                # Download the PDF
                response = requests.get(pdf_attachment.url).content
                pdf_content = ""
                
                # Read PDF content
                pdf_file = BytesIO(response)
                pdf_reader = PyPDF2.PdfReader(pdf_file)
                for page in pdf_reader.pages:
                    pdf_content += page.extract_text()
                
                # Append PDF content to message
                message = f"{message}\n\nPDF Content:\n{pdf_content}"
                
            except Exception as e:
                logger.error(f"Error processing PDF: {e}")
                await event.message.reply("sorry, i couldn't read that pdf")
                return
    
    # For non-music commands, use typing indicator and normal processing
    async with event.message.channel.typing:
        try:
            response = await asyncio.wait_for(
                generate_response(user_id, message, event),
                timeout=600.0
            )
            if isinstance(response, tuple):
                message_text, pdf_buffer = response
                file = interactions.File(pdf_buffer, file_name="practice_questions.pdf")
                message = await event.message.reply(message_text, files=[file])
            else:
                if len(response) >= 2000:
                    # Create a text file with the response
                    buffer = BytesIO(response.encode('utf-8'))
                    file = interactions.File(buffer, file_name="response.txt")
                    message = await event.message.reply("here's my response:", files=[file])
                else:
                    message = await event.message.reply(response[:1999])
            
            if event.message.author.id == 561968340072136719 or event.message.author.id == 404323034439352322:
                await message.add_reaction("💀")
                await message.add_reaction("😭")
        except asyncio.TimeoutError:
            await event.message.reply("sorry, that request took too long to process")

    # After handling the message, reset status back to default
    if (not event.message.author.bot and 
        event.message._mention_ids and 
        bot.user.id in event.message._mention_ids):
        
        await bot.change_presence(
            activity=interactions.Activity(
                name="no one rn", 
                type=interactions.ActivityType.LISTENING
            ),
            status=interactions.Status.DO_NOT_DISTURB
        )

@interactions.listen()
async def on_startup():
    global bot_voice_state
    print(f"Connected as {bot.user.display_name}")
    
    # Print server information
    print("\nServer List:")
    for guild in bot.guilds:
        try:
            # Get invite link for the guild
            invites = await guild.fetch_invites()
            invite_link = next((invite.url for invite in invites), "No invite found")
        except:
            invite_link = "Could not fetch invite"
            
        print(f"- {guild.name} (ID: {guild.id})")
        print(f"  Invite: {invite_link}")
        print(f"  Members: {guild.member_count}")
        print()
    
    load_course_info()
    load_conversation_histories()
    load_message_log()
    
    print("Setting presence...")
    await bot.change_presence(
        activity=interactions.Activity(
            name="no one rn", 
            type=interactions.ActivityType.LISTENING
        ),
        status=interactions.Status.DO_NOT_DISTURB
    )
    
    print("Joining voice channel...")
    voice_channel_id = 1275989030689046598
    try:
        voice_channel = await bot.fetch_channel(voice_channel_id)
        if not isinstance(voice_channel, interactions.GuildVoice):
            print(f"Channel {voice_channel_id} is not a voice channel")
            return
            
        print(f"Found channel: {voice_channel.name}")
        
        # Wait for bot to be ready before connecting
        await bot.wait_until_ready()
        
        # Connect to voice channel with a timeout
        try:
            bot_voice_state = await asyncio.wait_for(
                voice_channel.connect(),
                timeout=15.0
            )
            print(f"Connected to voice channel: {voice_channel.name}")
            
            # Wait a short time for the voice client to fully initialize
            await asyncio.sleep(2)
            
            if bot_voice_state and bot_voice_state.connected:
                print("Voice connection successful")
                # Start the queue processor only if connection is successful
                asyncio.create_task(process_music_queue(bot_voice_state))
                print("Started music queue processor")
            else:
                print("Voice connection failed to initialize properly")
                bot_voice_state = None
                
        except asyncio.TimeoutError:
            print("Voice connection timed out")
            bot_voice_state = None
        except Exception as e:
            print(f"Error connecting to voice: {e}")
            bot_voice_state = None
            
    except Exception as e:
        print(f"Failed to join voice channel: {e}")
        import traceback
        traceback.print_exc()
        bot_voice_state = None
    
    print("Starting token refresh task...")
    async def refresh_token():
        global token
        while True: 
            token = await get_spotify_token()
            await asyncio.sleep(3300) 
    
    asyncio.create_task(refresh_token())
    print("Startup complete")

if __name__ == "__main__":
    try:
        logger.info("Starting bot...")
        set_seed(1234)
        bot.start()
    except Exception as e:
        logger.error(f"Bot crashed: {e}")
