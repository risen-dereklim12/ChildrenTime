# Singapore Parents Bot

Singapore Parents Bot is a helper agent designed for families planning outings and activities in Singapore. It uses a Telegram interface powered by a language model (LLM) that acts as an agent. The bot can automatically check the weather, query local attractions, plan transit routes, and estimate bus arrivals depending on what you ask.

---

## Features

- **Current Weather:** Retrieves live weather conditions and forecasts across Singapore.
- **Attractions Search:** Searches a database of family-friendly attractions in Singapore, showing descriptions, location information, and ticket prices (including free options).
- **Directions and Transit:** Finds transit routes and gives instructions for traveling between two locations in Singapore.
- **Live Bus Arrivals:** Fetches real-time bus arrival timings for any bus stop code using the LTA DataMall API. If no LTA API key is provided, the bot falls back to simulated, realistic arrival times.

---

## Prerequisites

Before running the bot, you will need:

- **Python 3.10** or higher installed on your computer.
- A **Telegram Bot Token** which you can get by messaging [@BotFather](https://t.me/BotFather) on Telegram.
- An **LLM API Key** from your preferred provider. The bot supports OpenAI, Anthropic (Claude), Google Gemini, or any OpenAI-compatible API endpoint.

---

## Setup and Installation

### 1. Clone the repository
Move into the project folder:
```bash
cd ChildrenTime
```

### 2. Create and activate a virtual environment
Setting up a virtual environment keeps your project dependencies isolated.

On macOS and Linux:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

On Windows:
```cmd
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install the dependencies
Install the required packages using pip:
```bash
pip install -r requirements.txt
```

### 4. Set up your environment variables
You will need to configure your API keys and tokens. Copy the example environment file to create your own configuration file:

```bash
cp .env.example .env
```

Open the newly created `.env` file in a text editor and fill in your details:

```env
# Telegram Bot Configuration
TELEGRAM_BOT_TOKEN=your_telegram_bot_token_here

# LLM Configuration
# Set LLM_PROVIDER to: openai, gemini, anthropic, or openai-compatible
LLM_PROVIDER=anthropic
LLM_MODEL=claude-3-5-sonnet-20241022
LLM_API_KEY=your_llm_api_key_here

# LTA DataMall Configuration (Optional)
# If left blank, the bot will fall back to simulated bus arrival times.
LTA_ACCOUNT_KEY=your_lta_account_key_here
```

---

## Running the Bot

To start the bot, run the main Python script:

```bash
python bot.py
```

Once started, the console will print a message indicating it is listening for messages:
```text
Singapore Parents Bot is starting up...
Listening for messages... Press Ctrl+C to stop.
```

---

## Example Interactions

Find your bot on Telegram and type `/start` to see the greeting message. You can speak to it naturally. Here are some examples of what you can ask:

- *"What is the weather like right now?"*
- *"Show me free things to do with kids in Singapore"* or *"Tell me about Gardens by the Bay"*
- *"How do I get from Changi Airport to Marina Bay Sands?"*
- *"When is the next bus arriving at stop 01112?"*
