import os
import json
import random
import logging
import requests
import telebot
from dotenv import load_dotenv
from openai import OpenAI

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize API clients
telegram_token = os.getenv("TELEGRAM_BOT_TOKEN")
lta_key = os.getenv("LTA_ACCOUNT_KEY")

if not telegram_token:
    logger.warning("Missing TELEGRAM_BOT_TOKEN in environment. Bot may fail to start.")

bot = telebot.TeleBot(telegram_token) if telegram_token else None

class LLMClient:
    def __init__(self):
        self.provider = os.getenv("LLM_PROVIDER", "openai").lower()
        self.model = os.getenv("LLM_MODEL", "gpt-4o-mini")
        
        # Retrieve key
        self.api_key = os.getenv("LLM_API_KEY")
        if not self.api_key:
            # Fallbacks
            if self.provider == "openai":
                self.api_key = os.getenv("OPENAI_API_KEY")
            elif self.provider == "gemini":
                self.api_key = os.getenv("GEMINI_API_KEY")
            elif self.provider == "anthropic":
                self.api_key = os.getenv("ANTHROPIC_API_KEY")
                
        self.base_url = os.getenv("LLM_BASE_URL")
        
        # Configure OpenAI SDK client if applicable
        self.client = None
        if self.provider in ["openai", "openai-compatible", "gemini"]:
            actual_base_url = self.base_url
            if self.provider == "gemini" and not actual_base_url:
                actual_base_url = "https://generativelanguage.googleapis.com/v1beta/"
            
            if self.api_key:
                self.client = OpenAI(api_key=self.api_key, base_url=actual_base_url)
                
    def is_configured(self):
        if self.provider == "anthropic":
            return bool(self.api_key)
        return self.client is not None

    def chat_completion(self, messages, tools):
        if self.provider == "anthropic":
            return self._chat_completion_anthropic(messages, tools)
        else:
            return self._chat_completion_openai_compatible(messages, tools)

    def _chat_completion_openai_compatible(self, messages, tools):
        return self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=tools,
            tool_choice="auto"
        )

    def _chat_completion_anthropic(self, messages, tools):
        # Translate tools to Anthropic format
        anthropic_tools = []
        for t in tools:
            func = t["function"]
            anthropic_tools.append({
                "name": func["name"],
                "description": func["description"],
                "input_schema": func["parameters"]
            })
            
        # Translate messages
        system_prompt = ""
        anthropic_messages = []
        
        # Iterate over messages
        for m in messages:
            role = m.get("role")
            content = m.get("content")
            
            if role == "system":
                system_prompt += content + "\n"
            elif role == "user":
                anthropic_messages.append({"role": "user", "content": content})
            elif role == "assistant":
                if m.get("tool_calls"):
                    tool_uses = []
                    if content:
                        tool_uses.append({"type": "text", "text": content})
                    for tc in m["tool_calls"]:
                        tc_id = tc.get("id") if isinstance(tc, dict) else tc.id
                        tc_func = tc.get("function", {}) if isinstance(tc, dict) else tc.function
                        tc_name = tc_func.get("name") if isinstance(tc_func, dict) else tc_func.name
                        tc_args = tc_func.get("arguments") if isinstance(tc_func, dict) else tc_func.arguments
                        
                        tool_uses.append({
                            "type": "tool_use",
                            "id": tc_id,
                            "name": tc_name,
                            "input": json.loads(tc_args) if isinstance(tc_args, str) else tc_args
                        })
                    anthropic_messages.append({"role": "assistant", "content": tool_uses})
                else:
                    anthropic_messages.append({"role": "assistant", "content": content or ""})
            elif role == "tool":
                anthropic_messages.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": m.get("tool_call_id"),
                            "content": content
                        }
                    ]
                })
                
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }
        payload = {
            "model": self.model,
            "messages": anthropic_messages,
            "tools": anthropic_tools,
            "max_tokens": 1024
        }
        if system_prompt:
            payload["system"] = system_prompt.strip()
            
        logger.info(f"Sending Anthropic request (model={self.model})")
        resp = requests.post("https://api.anthropic.com/v1/messages", json=payload, headers=headers, timeout=30)
        
        if resp.status_code != 200:
            raise Exception(f"Anthropic API returned {resp.status_code}: {resp.text}")
            
        data = resp.json()
        
        # Translate response back to an OpenAI-like response object
        class ChoiceMessage:
            def __init__(self, content, tool_calls):
                self.role = "assistant"
                self.content = content
                self.tool_calls = tool_calls
                
        class Choice:
            def __init__(self, message):
                self.message = message
                
        class ToolCallFunction:
            def __init__(self, name, arguments):
                self.name = name
                self.arguments = arguments
                
        class ToolCall:
            def __init__(self, tc_id, name, arguments):
                self.id = tc_id
                self.type = "function"
                self.function = ToolCallFunction(name, arguments)
                
        class OpenAIResponse:
            def __init__(self, content, tool_calls):
                self.choices = [Choice(ChoiceMessage(content, tool_calls))]
                
        text_content = ""
        openai_tool_calls = []
        for block in data.get("content", []):
            if block.get("type") == "text":
                text_content += block.get("text", "")
            elif block.get("type") == "tool_use":
                tc_id = block.get("id")
                tc_name = block.get("name")
                tc_input = json.dumps(block.get("input", {}))
                openai_tool_calls.append(ToolCall(tc_id, tc_name, tc_input))
                
        return OpenAIResponse(text_content or None, openai_tool_calls or None)

llm_client = LLMClient()

# --- WMO Weather Codes mapping ---
WMO_CODES = {
    0: "Clear sky ☀️",
    1: "Mainly clear 🌤️", 2: "Partly cloudy ⛅", 3: "Overcast ☁️",
    45: "Foggy 🌫️", 48: "Depositing rime fog 🌫️",
    51: "Light drizzle 🌧️", 53: "Moderate drizzle 🌧️", 55: "Dense drizzle 🌧️",
    61: "Slight rain 🌧️", 63: "Moderate rain 🌧️", 65: "Heavy rain 🌧️",
    80: "Slight rain showers 🌦️", 81: "Moderate rain showers 🌦️", 82: "Violent rain showers ⛈️",
    95: "Thunderstorm ⛈️", 96: "Thunderstorm with slight hail ⛈️", 99: "Thunderstorm with heavy hail ⛈️"
}

# --- Singapore Attractions Database ---
ATTRACTIONS = [
    {
        "name": "Gardens by the Bay",
        "description": "A futuristic park featuring giant Supertrees, greenhouse conservatories (Flower Dome & Cloud Forest), and light shows.",
        "location": "Marina Bay (Bayfront MRT)",
        "prices": {
            "adult": "$53 (Double Conservatories)",
            "child": "$40 (3-12 years)",
            "outdoor_gardens": "Free"
        },
        "opening_hours": "9:00 AM - 9:00 PM daily",
        "category": "Nature/Park"
    },
    {
        "name": "Universal Studios Singapore (USS)",
        "description": "A world-class movie theme park located within Resorts World Sentosa, featuring rides, shows, and attractions based on popular films.",
        "location": "Sentosa (HarbourFront MRT -> Sentosa Express)",
        "prices": {
            "adult": "$83",
            "child": "$62 (4-12 years)"
        },
        "opening_hours": "10:00 AM - 7:00 PM daily",
        "category": "Theme Park/Family"
    },
    {
        "name": "Night Safari",
        "description": "The world's first nocturnal zoo, offering a tram ride and walking trails to see animals in naturalistic nighttime habitats.",
        "location": "Mandai (Khatib MRT -> Mandai Shuttle)",
        "prices": {
            "adult": "$55",
            "child": "$38 (3-12 years)"
        },
        "opening_hours": "7:15 PM - 12:00 AM daily",
        "category": "Wildlife/Zoo"
    },
    {
        "name": "Singapore Zoo",
        "description": "A world-renowned open-concept rainforest zoo known for its lush vegetation, animal shows, and interactive feeding sessions.",
        "location": "Mandai (Khatib MRT -> Mandai Shuttle)",
        "prices": {
            "adult": "$49",
            "child": "$34 (3-12 years)"
        },
        "opening_hours": "8:30 AM - 6:00 PM daily",
        "category": "Wildlife/Zoo"
    },
    {
        "name": "Singapore Flyer",
        "description": "A giant observation wheel offering panoramic 360-degree views of the Singapore skyline and Marina Bay.",
        "location": "Marina Centre (Promenade MRT)",
        "prices": {
            "adult": "$40",
            "child": "$25 (3-12 years)"
        },
        "opening_hours": "10:00 AM - 10:00 PM daily",
        "category": "Sightseeing/Views"
    },
    {
        "name": "S.E.A. Aquarium",
        "description": "One of the world's largest aquariums, home to more than 100,000 marine animals across 40 diverse habitats.",
        "location": "Sentosa (HarbourFront MRT -> Sentosa Express)",
        "prices": {
            "adult": "$43",
            "child": "$32 (4-12 years)"
        },
        "opening_hours": "10:00 AM - 5:00 PM daily",
        "category": "Wildlife/Aquarium"
    },
    {
        "name": "Marina Bay Sands SkyPark Observation Deck",
        "description": "A large wooden observation deck offering breathtaking vistas of Marina Bay, the city skyline, and the Gardens by the Bay.",
        "location": "Marina Bay (Bayfront MRT)",
        "prices": {
            "adult": "$32",
            "child": "$28 (2-12 years)"
        },
        "opening_hours": "11:00 AM - 9:00 PM daily",
        "category": "Sightseeing/Views"
    },
    {
        "name": "Jewel Changi Airport",
        "description": "A nature-themed entertainment and retail complex inside Changi Airport, featuring the world's tallest indoor waterfall (Rain Vortex).",
        "location": "Changi Airport (Changi Airport MRT)",
        "prices": {
            "rain_vortex": "Free",
            "canopy_park_admission": "$8 (Adult/Child)"
        },
        "opening_hours": "24 Hours (Rain Vortex: 11 AM - 10 PM; Canopy Park: 10 AM - 10 PM)",
        "category": "Sightseeing/Shopping"
    },
    {
        "name": "Merlion Park",
        "description": "A popular landmark featuring the iconic Merlion statue spitting water into Marina Bay. Great for photos.",
        "location": "Downtown (Raffles Place MRT)",
        "prices": {
            "admission": "Free"
        },
        "opening_hours": "24 Hours",
        "category": "Sightseeing/Landmark"
    }
]

# --- Pre-mapped Transit Routes ---
TRANSIT_ROUTES = {
    ("gardens by the bay", "universal studios singapore"): (
        "Take the MRT Circle Line (Yellow) from Bayfront MRT to HarbourFront MRT (5 stops, approx. 10 mins).\n"
        "From HarbourFront, head to VivoCity Level 3 and take the Sentosa Express monorail to Waterfront Station (approx. 5 mins).\n"
        "Alternatively, you can walk across the Sentosa Boardwalk from VivoCity (approx. 15 mins, free admission).\n"
        "Estimated cost: ~$2.50. Total travel time: ~20 mins."
    ),
    ("universal studios singapore", "gardens by the bay"): (
        "Take the Sentosa Express monorail from Waterfront Station to VivoCity (HarbourFront MRT).\n"
        "Take the Circle Line (Yellow) from HarbourFront MRT to Bayfront MRT (5 stops, approx. 10 mins).\n"
        "Estimated cost: ~$2.50. Total travel time: ~20 mins."
    ),
    ("changi airport", "gardens by the bay"): (
        "Take the East-West Line (Green) from Changi Airport MRT to Tanah Merah MRT (2 stops).\n"
        "Cross the platform and take the East-West Line to Expo MRT, then transfer to the Downtown Line (Blue) directly to Bayfront MRT.\n"
        "Estimated cost: ~$2.10. Total travel time: ~50 mins."
    ),
    ("gardens by the bay", "changi airport"): (
        "Take the Downtown Line (Blue) from Bayfront MRT to Expo MRT.\n"
        "Transfer to the East-West Line (Green) toward Tanah Merah MRT, then take the Airport line to Changi Airport MRT.\n"
        "Estimated cost: ~$2.10. Total travel time: ~50 mins."
    ),
    ("changi airport", "universal studios singapore"): (
        "Take the East-West Line (Green) from Changi Airport MRT to Tanah Merah MRT (2 stops).\n"
        "Cross the platform and take the East-West Line to Outram Park MRT, then transfer to the North-East Line (Purple) to HarbourFront MRT.\n"
        "From HarbourFront, head to VivoCity Level 3 and take the Sentosa Express to Waterfront Station.\n"
        "Estimated cost: ~$2.60. Total travel time: ~70 mins."
    ),
    ("universal studios singapore", "changi airport"): (
        "Take the Sentosa Express monorail from Waterfront Station to VivoCity (HarbourFront MRT).\n"
        "Take the North-East Line (Purple) from HarbourFront MRT to Outram Park MRT.\n"
        "Transfer to the East-West Line (Green) to Tanah Merah MRT, then take the airport train to Changi Airport MRT.\n"
        "Estimated cost: ~$2.60. Total travel time: ~70 mins."
    ),
    ("merlion park", "gardens by the bay"): (
        "Walk from Merlion Park across the Jubilee Bridge/Helix Bridge to Gardens by the Bay (approx. 20 mins, highly scenic walk).\n"
        "Alternatively, take the Downtown Line (Blue) from Raffles Place MRT (walk 5 mins) to Bayfront MRT (1 stop).\n"
        "Estimated cost: ~$1.40. Total travel time: ~10 mins."
    ),
    ("gardens by the bay", "merlion park"): (
        "Take the Downtown Line (Blue) from Bayfront MRT to Raffles Place MRT (1 stop, walk 5 mins to Merlion Park).\n"
        "Or take the scenic walk across Helix Bridge (approx. 20 mins).\n"
        "Estimated cost: ~$1.40. Total travel time: ~10 mins."
    ),
    ("singapore zoo", "gardens by the bay"): (
        "Take the Mandai Khatib Shuttle (bus) from Singapore Zoo to Khatib MRT Station (~15 mins).\n"
        "Take the North-South Line (Red) from Khatib MRT to Newton MRT, then transfer to the Downtown Line (Blue) to Bayfront MRT.\n"
        "Estimated cost: ~$3.00 (including shuttle). Total travel time: ~60 mins."
    ),
    ("gardens by the bay", "singapore zoo"): (
        "Take the Downtown Line (Blue) from Bayfront MRT to Newton MRT, transfer to the North-South Line (Red) to Khatib MRT.\n"
        "From Khatib MRT, take the Mandai Khatib Shuttle directly to Singapore Zoo (runs every 10 mins, cost $1).\n"
        "Estimated cost: ~$3.00. Total travel time: ~60 mins."
    )
}

# --- Popular Singapore Landmarks for Location Matching ---
LANDMARKS = {
    "Changi Airport": (1.3644, 103.9915),
    "Gardens by the Bay": (1.2816, 103.8636),
    "Universal Studios Singapore": (1.2540, 103.8238),
    "Singapore Zoo": (1.4043, 103.7930),
    "Merlion Park": (1.2868, 103.8545),
    "Orchard Road": (1.3048, 103.8318),
    "Marina Bay Sands": (1.2823, 103.8585),
    "Sentosa Island": (1.2494, 103.8303),
    "Jurong Bird Park / Jurong Lake Gardens": (1.3392, 103.7058),
    "East Coast Park": (1.3008, 103.9126),
    "Raffles Place MRT": (1.2839, 103.8515),
    "City Hall MRT": (1.2929, 103.8526),
    "Dhoby Ghaut MRT": (1.2989, 103.8462),
    "Outram Park MRT": (1.2801, 103.8394),
    "Bugis MRT": (1.3007, 103.8561),
    "Woodlands MRT": (1.4368, 103.7865),
    "Jurong East MRT": (1.3331, 103.7421),
    "Tampines MRT": (1.3533, 103.9452),
    "Serangoon MRT": (1.3506, 103.8728),
    "Bishan MRT": (1.3508, 103.8497)
}

def get_nearest_landmark(lat, lon):
    """Find the closest pre-mapped Singapore landmark using the Haversine formula."""
    import math
    min_dist = float('inf')
    closest_name = "Singapore Central"
    
    for name, coords in LANDMARKS.items():
        lat1, lon1 = math.radians(lat), math.radians(lon)
        lat2, lon2 = math.radians(coords[0]), math.radians(coords[1])
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        r = 6371 # Radius of Earth in kilometers
        dist = c * r
        
        if dist < min_dist:
            min_dist = dist
            closest_name = name
            
    return closest_name

# --- Tool Implementations ---

def get_current_weather():
    """Get the current weather conditions for Singapore."""
    logger.info("Tool executed: get_current_weather")
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": 1.3521,
        "longitude": 103.8198,
        "current": "temperature_2m,relative_humidity_2m,weather_code,wind_speed_10m",
        "timezone": "Asia/Singapore"
    }
    try:
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            data = response.json()
            curr = data.get("current", {})
            temp = curr.get("temperature_2m")
            humidity = curr.get("relative_humidity_2m")
            code = curr.get("weather_code")
            wind = curr.get("wind_speed_10m")
            
            desc = WMO_CODES.get(code, "Unknown weather conditions ❓")
            return json.dumps({
                "status": "success",
                "temperature": f"{temp}°C",
                "humidity": f"{humidity}%",
                "wind_speed": f"{wind} km/h",
                "description": desc
            })
        else:
            return json.dumps({"status": "error", "message": f"HTTP {response.status_code} from Weather Service."})
    except Exception as e:
        logger.error(f"Error fetching weather: {e}")
        return json.dumps({"status": "error", "message": str(e)})


def search_attractions(query=None, category=None):
    """Search for attractions by keyword or category."""
    logger.info(f"Tool executed: search_attractions (query={query}, category={category})")
    results = []
    for att in ATTRACTIONS:
        match = True
        if category:
            if category.lower() not in att["category"].lower():
                match = False
        if query:
            q = query.lower()
            in_name = q in att["name"].lower()
            in_desc = q in att["description"].lower()
            in_loc = q in att["location"].lower()
            in_cat = q in att["category"].lower()
            if not (in_name or in_desc or in_loc or in_cat):
                match = False
        if match:
            results.append(att)
    return json.dumps({"status": "success", "count": len(results), "attractions": results})


def get_transit_route(origin, destination):
    """Get transit route directions between popular locations in Singapore."""
    logger.info(f"Tool executed: get_transit_route (from={origin}, to={destination})")
    origin_clean = origin.lower().strip()
    destination_clean = destination.lower().strip()
    
    # Search for matching routes in pre-defined table
    for (o, d), route in TRANSIT_ROUTES.items():
        if (o in origin_clean or origin_clean in o) and (d in destination_clean or destination_clean in d):
            return json.dumps({
                "status": "success",
                "origin": origin,
                "destination": destination,
                "directions": route
            })
            
    # General fallback instructions
    directions = (
        f"To travel from '{origin}' to '{destination}' in Singapore:\n"
        "1. Board the nearest MRT train. Check the MRT transit map for transfer stations.\n"
        "2. If travelling to popular spots like Mandai (Zoo/Night Safari), exit at Khatib MRT and board the Mandai Shuttle.\n"
        "3. For customized public bus or train routes, we recommend searching 'Singapore transit directions' or using OneMap/Google Maps routing."
    )
    return json.dumps({
        "status": "partial_match",
        "origin": origin,
        "destination": destination,
        "directions": directions
    })


def get_bus_arrival(bus_stop_code, service_no=None):
    """Get estimated arrival times for buses at a specific Singapore bus stop code."""
    logger.info(f"Tool executed: get_bus_arrival (stop={bus_stop_code}, service={service_no})")
    
    # Try calling LTA API if AccountKey is provided
    if lta_key:
        url = "https://datamall2.mytransport.sg/ltaodataservice/BusArrivalv2"
        params = {"BusStopCode": bus_stop_code}
        headers = {"AccountKey": lta_key, "accept": "application/json"}
        try:
            response = requests.get(url, params=params, headers=headers, timeout=10)
            if response.status_code == 200:
                data = response.json()
                services = data.get("Services", [])
                
                # Filter by service number if requested
                if service_no:
                    services = [s for s in services if s.get("ServiceNo") == str(service_no)]
                    
                results = []
                for s in services:
                    sv_no = s.get("ServiceNo")
                    nxt = s.get("NextBus", {})
                    nxt2 = s.get("NextBus2", {})
                    
                    # Convert LTA ETA string to minutes
                    # ETA format: 2026-07-08T19:15:30+08:00
                    # For simplicity, returning raw LTA response or basic mapping
                    results.append({
                        "service_no": sv_no,
                        "next_bus": {
                            "estimated_arrival": nxt.get("EstimatedArrival"),
                            "load": nxt.get("Load"), # SEA (Seats Available), SDA (Standing Available), LSD (Limited Standing)
                            "feature": nxt.get("Feature") # WAB (Wheelchair Accessible)
                        },
                        "following_bus": {
                            "estimated_arrival": nxt2.get("EstimatedArrival"),
                            "load": nxt2.get("Load")
                        }
                    })
                return json.dumps({"status": "success", "bus_stop_code": bus_stop_code, "services": results})
        except Exception as e:
            logger.warning(f"Failed to fetch real-time LTA data, falling back to mock: {e}")
            
    # Mock Fallback if no LTA key or request fails
    logger.info("Returning simulated bus arrival data")
    mock_services = ["166", "147", "197", "851", "961"]
    if service_no:
        services_to_mock = [service_no]
    else:
        services_to_mock = random.sample(mock_services, k=min(3, len(mock_services)))
        
    results = []
    for s in services_to_mock:
        next_min = random.randint(1, 8)
        nxt2_min = next_min + random.randint(5, 12)
        results.append({
            "service_no": s,
            "next_bus": {
                "estimated_arrival_in_minutes": f"{next_min} mins",
                "load": random.choice(["Seats Available 🟢", "Standing Available 🟡", "Limited Standing 🔴"]),
                "wheelchair_accessible": random.choice([True, False])
            },
            "following_bus": {
                "estimated_arrival_in_minutes": f"{nxt2_min} mins",
                "load": random.choice(["Seats Available 🟢", "Standing Available 🟡"])
            }
        })
    return json.dumps({
        "status": "mocked",
        "bus_stop_code": bus_stop_code,
        "services": results,
        "note": "This is simulated real-time data since no LTA Account Key was configured."
    })


# --- OpenAI Agent Orchestration ---

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather conditions (temperature, humidity, description) for Singapore.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_attractions",
            "description": "Search Singapore tourist attractions, descriptions, opening hours, categories, and admission ticket prices.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Keyword to search in attraction names, locations, categories or descriptions (e.g. 'zoo', 'admission price', 'free')."
                    },
                    "category": {
                        "type": "string",
                        "description": "Optional category filter (e.g., 'Wildlife/Zoo', 'Nature/Park', 'Theme Park/Family', 'Sightseeing/Views')."
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_transit_route",
            "description": "Get transit directions (MRT/bus trains) between popular tourist attractions or stations in Singapore.",
            "parameters": {
                "type": "object",
                "properties": {
                    "origin": {
                        "type": "string",
                        "description": "The starting attraction name or location (e.g., 'Changi Airport', 'Gardens by the Bay')."
                    },
                    "destination": {
                        "type": "string",
                        "description": "The destination attraction name or location (e.g., 'Universal Studios Singapore', 'Singapore Zoo')."
                    }
                },
                "required": ["origin", "destination"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_bus_arrival",
            "description": "Get real-time (or simulated) estimated bus arrival times, occupancy loads, and accessibility for a Singapore bus stop code.",
            "parameters": {
                "type": "object",
                "properties": {
                    "bus_stop_code": {
                        "type": "string",
                        "description": "The 5-digit bus stop code to query (e.g., '01112')."
                    },
                    "service_no": {
                        "type": "string",
                        "description": "Optional specific bus number (e.g. '166', '147') to filter the result."
                    }
                },
                "required": ["bus_stop_code"]
            }
        }
    }
]

def run_agent(user_message, chat_history=None, chat_id=None):
    if not llm_client.is_configured():
        return (
            f"⚠️ LLM provider '{llm_client.provider}' is not configured properly. "
            "Please check that you have defined the appropriate API key in your .env file."
        )
        
    if chat_history is None:
        chat_history = []
        
    system_prompt = (
        "You are a helpful, professional travel and family outing agent for Singapore. "
        "You can check current weather, search attractions with descriptions & prices, "
        "provide MRT/bus routes between popular points, and query bus arrival times at bus stops. "
        "Always make sure to explain admission prices clearly if requested. "
        "If the user asks about travelling somewhere, check if your transit route tool has directions. "
        "Keep responses engaging, concise, and structured with clear markdown formatting. "
        "Suggest next steps or other attractions/transport queries where appropriate."
    )
    
    # Inject user's shared location if available
    if chat_id and chat_id in user_locations:
        loc = user_locations[chat_id]
        nearest = get_nearest_landmark(loc["latitude"], loc["longitude"])
        system_prompt += (
            f"\n\nCURRENT USER LOCATION: The user has shared their live location: "
            f"Latitude {loc['latitude']}, Longitude {loc['longitude']}. They are currently near {nearest}. "
            f"If they ask for directions or travel options from 'here', 'my location', 'where I am', etc., "
            f"assume their starting point (origin) is '{nearest}'."
        )
    else:
        system_prompt += (
            "\n\nCURRENT USER LOCATION: The user's current location is unknown. "
            "If they ask for directions from 'here' or 'my location', you must politely ask them "
            "to share their location using the Telegram attachment button so you can help them."
        )
        
    messages = [
        {
            "role": "system",
            "content": system_prompt
        }
    ]
    
    # Load recent conversation history (last 10 messages)
    messages.extend(chat_history[-10:])
    messages.append({"role": "user", "content": user_message})
    
    available_tools = {
        "get_current_weather": get_current_weather,
        "search_attractions": search_attractions,
        "get_transit_route": get_transit_route,
        "get_bus_arrival": get_bus_arrival
    }
    
    # Execution loop
    for _ in range(5):
        try:
            response = llm_client.chat_completion(
                messages=messages,
                tools=TOOLS
            )
        except Exception as e:
            logger.error(f"LLM API Call Failed: {e}")
            return f"❌ Sorry, I encountered an error communicating with the AI service: {str(e)}"
            
        res_msg = response.choices[0].message
        
        # Translate to plain dict before appending to keep context representation uniform
        res_dict = {
            "role": "assistant",
            "content": res_msg.content,
        }
        if res_msg.tool_calls:
            res_dict["tool_calls"] = []
            for tc in res_msg.tool_calls:
                res_dict["tool_calls"].append({
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments
                    }
                })
        messages.append(res_dict)
        
        if res_msg.tool_calls:
            for tool_call in res_msg.tool_calls:
                func_name = tool_call.function.name
                func_args = json.loads(tool_call.function.arguments)
                
                if func_name in available_tools:
                    tool_func = available_tools[func_name]
                    try:
                        tool_out = tool_func(**func_args)
                    except Exception as err:
                        tool_out = json.dumps({"status": "error", "message": str(err)})
                    
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": func_name,
                        "content": tool_out
                    })
                else:
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": func_name,
                        "content": json.dumps({"status": "error", "message": "Tool not found"})
                    })
        else:
            # We got a final text answer
            return res_msg.content
            
    return "⚠️ I couldn't complete the query reasoning loop in a reasonable number of steps. Please try again with a simpler question."


# --- Telegram Bot Handler ---

chat_histories = {} # memory store for chat sessions: chat_id -> list of message dicts
user_locations = {} # store for user locations: chat_id -> {"latitude": lat, "longitude": lon}

if bot:
    @bot.message_handler(commands=['start', 'help'])
    def send_welcome(message):
        welcome_text = (
            "🇸🇬 *Welcome to the Singapore Parents Bot!* 🇸🇬\n\n"
            "I'm here to help you plan your family trips in Singapore! You can ask me:\n"
            "🌤️ *Weather:* \"What's the weather like right now?\"\n"
            "🎡 *Attractions & Prices:* \"Tell me about Gardens by the Bay\" or \"Which attractions are free?\"\n"
            "🚇 *Directions:* \"How do I get from Changi Airport to MBS?\" or \"How do I get to Singapore Zoo from here?\"\n"
            "🚌 *Bus Arrivals:* \"When is the next bus arriving at stop 01112?\"\n\n"
            "📍 *Tip:* You can share your current location using the attachment button at any time, and I'll use it to give you directions from \"here\"!"
        )
        bot.reply_to(message, welcome_text, parse_mode="Markdown")

    @bot.message_handler(content_types=['location'])
    def handle_location(message):
        chat_id = message.chat.id
        lat = message.location.latitude
        lon = message.location.longitude
        user_locations[chat_id] = {"latitude": lat, "longitude": lon}
        
        nearest = get_nearest_landmark(lat, lon)
        reply_text = (
            f"📍 *Location Saved!*\n\n"
            f"I see you are near *{nearest}*. "
            f"Now you can ask me things like: \"How do I get to Gardens by the Bay from here?\""
        )
        bot.reply_to(message, reply_text, parse_mode="Markdown", reply_markup=telebot.types.ReplyKeyboardRemove())

    @bot.message_handler(func=lambda message: True)
    def handle_user_message(message):
        chat_id = message.chat.id
        user_text = message.text
        
        # Show "typing..." status while processing
        bot.send_chat_action(chat_id, 'typing')
        
        # Retrieve history
        if chat_id not in chat_histories:
            chat_histories[chat_id] = []
            
        history = chat_histories[chat_id]
        
        # Get response from LLM Agent
        agent_reply = run_agent(user_text, history, chat_id=chat_id)
        
        # Save to history
        history.append({"role": "user", "content": user_text})
        history.append({"role": "assistant", "content": agent_reply})
        
        # Keep history to last 10 messages to manage context size
        if len(history) > 10:
            chat_histories[chat_id] = history[-10:]
            
        # Determine if we should show the location request button
        reply_markup = None
        prompt_keywords = ["share your location", "provide your location", "send your location", "send me your location", "where you are"]
        if any(kw in agent_reply.lower() for kw in prompt_keywords):
            reply_markup = telebot.types.ReplyKeyboardMarkup(one_time_keyboard=True, resize_keyboard=True)
            reply_markup.add(telebot.types.KeyboardButton("📍 Share Location", request_location=True))
            
        # Send reply
        try:
            bot.reply_to(message, agent_reply, parse_mode="Markdown", reply_markup=reply_markup)
        except Exception as e:
            # Fallback if markdown parsing fails
            logger.warning(f"Failed to send with Markdown, trying plain text: {e}")
            bot.reply_to(message, agent_reply, reply_markup=reply_markup)


# --- Main Entry Point ---

if __name__ == "__main__":
    if not telegram_token or not llm_client.is_configured():
        print("❌ Error: TELEGRAM_BOT_TOKEN and LLM API credentials must be set in your .env file!")
    else:
        print("🇸🇬 Singapore Travel Agent Telegram Bot is starting up...")
        print("Listening for messages... Press Ctrl+C to stop.")
        try:
            bot.infinity_polling()
        except KeyboardInterrupt:
            print("\nStopping bot...")
