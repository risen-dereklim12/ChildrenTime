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
    "Serangoon MRT": (1.3506, 103.8728),
    "Bishan MRT": (1.3508, 103.8497)
}


class OneMapAuth:
    """Manages the generation, validation, and in-memory caching of the OneMap JWT access token."""
    def __init__(self):
        self.token = None
        self.expires_at = 0

    def get_token(self):
        import time
        now = time.time()
        
        # 1. Use cached token if still valid
        if self.token and self.expires_at > now + 300:
            return self.token

        email = os.getenv("ONEMAP_EMAIL")
        password = os.getenv("ONEMAP_PASSWORD")
        sdk_key = os.getenv("LTA_SDK_KEY")

        # 2. Check if the user accidentally put a JWT token in password or sdk_key
        for key_candidate in [password, sdk_key]:
            if key_candidate and (len(key_candidate) > 100 or key_candidate.startswith("eyJ")):
                return key_candidate

        # 3. Request a new token dynamically using credentials
        if email and password:
            try:
                url = "https://www.onemap.gov.sg/api/auth/post/getToken"
                payload = {"email": email, "password": password}
                response = requests.post(url, json=payload, timeout=10)
                if response.status_code == 200:
                    res_data = response.json()
                    token = res_data.get("access_token")
                    if token:
                        self.token = token
                        self.expires_at = now + 250000
                        logger.info("Generated new OneMap access token using credentials.")
                        return token
                logger.warning(f"OneMap getToken API returned status {response.status_code}: {response.text}")
            except Exception as e:
                logger.error(f"Error fetching token from OneMap: {e}")

        # 4. Fallback to manually configured LTA_SDK_KEY if it's a JWT
        if sdk_key and len(sdk_key) > 50:
            return sdk_key

        return None


class WeatherService:
    """Encapsulates weather forecasts and current conditions retrieval from Open-Meteo."""
    def get_current_weather(self):
        logger.info("WeatherService: Fetching current weather")
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
            logger.error(f"WeatherService error: {e}")
            return json.dumps({"status": "error", "message": str(e)})


class TransitService:
    """Encapsulates geocoding, public transport routing, and formatters via OneMap APIs."""
    def __init__(self, auth_provider: OneMapAuth):
        self.auth = auth_provider

    def resolve_to_coords(self, loc_str):
        loc_clean = loc_str.lower().strip()
        
        # Try parsing as coordinates
        try:
            parts = loc_clean.split(",")
            if len(parts) == 2:
                lat = float(parts[0].strip())
                lon = float(parts[1].strip())
                return lat, lon
        except ValueError:
            pass
            
        sdk_key = self.auth.get_token()
        if sdk_key:
            try:
                url = "https://www.onemap.gov.sg/api/common/elastic/search"
                params = {
                    "searchVal": loc_str,
                    "returnGeom": "Y",
                    "getAddrDetails": "N",
                    "pageNum": 1
                }
                headers = {"Authorization": sdk_key}
                if len(sdk_key) > 50 and not sdk_key.startswith("Bearer "):
                    headers["Authorization"] = f"Bearer {sdk_key}"
                    
                response = requests.get(url, params=params, headers=headers, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    results = data.get("results", [])
                    if results:
                        first_res = results[0]
                        lat = float(first_res["LATITUDE"])
                        lon = float(first_res["LONGITUDE"])
                        logger.info(f"Resolved location '{loc_str}' dynamically via OneMap Search API: ({lat}, {lon})")
                        return lat, lon
            except Exception as e:
                logger.error(f"TransitService geocoding error for '{loc_str}': {e}")

        # Match against known landmarks as fallback
        for name, coords in LANDMARKS.items():
            if name.lower() in loc_clean or loc_clean in name.lower():
                return coords
                
        return None

    def format_pt_route(self, response_json):
        if "plan" not in response_json or "itineraries" not in response_json["plan"] or not response_json["plan"]["itineraries"]:
            return "No public transport route found between the specified locations."
            
        itinerary = response_json["plan"]["itineraries"][0]
        total_time = round(itinerary.get("duration", 0) / 60)
        legs = itinerary.get("legs", [])
        
        directions = []
        for i, leg in enumerate(legs):
            mode = leg.get("mode", "UNKNOWN")
            duration = round(leg.get("duration", 0) / 60)
            from_name = leg.get("from", {}).get("name", "Origin")
            to_name = leg.get("to", {}).get("name", "Destination")
            
            if from_name.replace(".", "").replace(",", "").replace("-", "").isdigit():
                from_name = "your location"
                
            if mode == "WALK":
                distance = round(leg.get("distance", 0))
                directions.append(f"{i+1}. Walk from {from_name} to {to_name} (approx. {distance}m, {duration} mins)")
            elif mode == "BUS":
                bus_num = leg.get("route", "")
                directions.append(f"{i+1}. Board Bus {bus_num} at {from_name} and ride to {to_name} ({duration} mins)")
            elif mode in ["SUBWAY", "RAIL"]:
                line_name = leg.get("route", "MRT")
                directions.append(f"{i+1}. Board the MRT ({line_name}) from {from_name} to {to_name} ({duration} mins)")
            else:
                directions.append(f"{i+1}. Take {mode} from {from_name} to {to_name} ({duration} mins)")
                
        return f"Total travel time: ~{total_time} mins.\n\nRoute steps:\n" + "\n".join(directions)

    def get_transit_route(self, origin, destination):
        logger.info(f"TransitService: Routing from '{origin}' to '{destination}'")
        origin_coords = self.resolve_to_coords(origin)
        dest_coords = self.resolve_to_coords(destination)
        
        sdk_key = self.auth.get_token()
        if origin_coords and dest_coords and sdk_key:
            try:
                import datetime
                now = datetime.datetime.now()
                url = "https://www.onemap.gov.sg/api/public/routingsvc/route"
                params = {
                    "start": f"{origin_coords[0]},{origin_coords[1]}",
                    "end": f"{dest_coords[0]},{dest_coords[1]}",
                    "routeType": "pt",
                    "date": now.strftime("%m-%d-%Y"),
                    "time": now.strftime("%H:%M:%S"),
                    "mode": "transit"
                }
                headers = {"Authorization": sdk_key}
                if len(sdk_key) > 50 and not sdk_key.startswith("Bearer "):
                    headers["Authorization"] = f"Bearer {sdk_key}"
                    
                response = requests.get(url, params=params, headers=headers, timeout=10)
                if response.status_code == 200:
                    res_data = response.json()
                    route_desc = self.format_pt_route(res_data)
                    return json.dumps({
                        "status": "success",
                        "origin": origin,
                        "destination": destination,
                        "directions": route_desc
                    })
                else:
                    logger.warning(f"OneMap Routing API status {response.status_code}: {response.text}")
            except Exception as e:
                logger.error(f"TransitService routing error: {e}")
                
        # Fallback offline directions
        directions = (
            f"To travel from '{origin}' to '{destination}' in Singapore:\n"
            "1. Board the nearest MRT train. Check the MRT transit map for transfer stations.\n"
            "2. If travelling to popular spots like Mandai (Zoo/Night Safari), exit at Khatib MRT and board the Mandai Shuttle.\n"
            "3. For customized public bus or train routes, we recommend searching 'Singapore transit directions' or using OneMap/Google Maps routing."
        )
        if sdk_key:
            directions = "⚠️ Note: Live routing API query failed. Displaying default offline directions:\n\n" + directions
        else:
            directions = (
                "⚠️ Note: OneMap authentication failed or credentials are not configured. "
                "To enable live transit routing, please register a free account at onemap.gov.sg and "
                "add ONEMAP_EMAIL and ONEMAP_PASSWORD to your .env file.\n\n"
                "Displaying default offline directions:\n\n" + directions
            )
        return json.dumps({
            "status": "partial_match",
            "origin": origin,
            "destination": destination,
            "directions": directions
        })

    def get_nearest_landmark(self, lat, lon):
        sdk_key = self.auth.get_token()
        if sdk_key:
            try:
                url = "https://www.onemap.gov.sg/api/public/revgeocode"
                params = {
                    "location": f"{lat},{lon}",
                    "buffer": 100,
                    "addressType": "All"
                }
                headers = {"Authorization": sdk_key}
                if len(sdk_key) > 50 and not sdk_key.startswith("Bearer "):
                    headers["Authorization"] = f"Bearer {sdk_key}"
                    
                response = requests.get(url, params=params, headers=headers, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    geocode_info = data.get("GeocodeInfo", [])
                    if geocode_info:
                        # Extract the first address found
                        first_info = geocode_info[0]
                        building = first_info.get("BUILDING")
                        road = first_info.get("ROAD")
                        block = first_info.get("BLOCK")
                        
                        # Return building name if valid, otherwise build address from block + road
                        if building and building != "NIL" and building != "null":
                            return building.title()
                        
                        addr_parts = []
                        if block and block != "NIL" and block != "null":
                            addr_parts.append(block)
                        if road and road != "NIL" and road != "null":
                            addr_parts.append(road.title())
                            
                        if addr_parts:
                            return " ".join(addr_parts)
            except Exception as e:
                logger.error(f"TransitService reverse geocoding error: {e}")

        # Offline fallback using Haversine formula
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


class BusArrivalService:
    """Encapsulates bus arrival timings lookup using the LTA DataMall API."""
    def __init__(self, lta_key=None):
        self.lta_key = lta_key or os.getenv("LTA_ACCOUNT_KEY")

    def get_bus_arrival(self, bus_stop_code, service_no=None):
        logger.info(f"BusArrivalService: Querying stop={bus_stop_code}, service={service_no}")
        if self.lta_key:
            url = "https://datamall2.mytransport.sg/ltaodataservice/BusArrivalv2"
            params = {"BusStopCode": bus_stop_code}
            headers = {"AccountKey": self.lta_key, "accept": "application/json"}
            try:
                response = requests.get(url, params=params, headers=headers, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    services = data.get("Services", [])
                    if service_no:
                        services = [s for s in services if s.get("ServiceNo") == str(service_no)]
                        
                    results = []
                    for s in services:
                        sv_no = s.get("ServiceNo")
                        nxt = s.get("NextBus", {})
                        nxt2 = s.get("NextBus2", {})
                        results.append({
                            "service_no": sv_no,
                            "next_bus": {
                                "estimated_arrival": nxt.get("EstimatedArrival"),
                                "load": nxt.get("Load"),
                                "feature": nxt.get("Feature")
                            },
                            "following_bus": {
                                "estimated_arrival": nxt2.get("EstimatedArrival"),
                                "load": nxt2.get("Load")
                            }
                        })
                    return json.dumps({"status": "success", "bus_stop_code": bus_stop_code, "services": results})
            except Exception as e:
                logger.warning(f"Failed to fetch real-time LTA data: {e}")
                
        # Simulated fallback
        logger.info("Returning simulated bus arrival data")
        mock_services = ["166", "147", "197", "851", "961"]
        services_to_mock = [service_no] if service_no else random.sample(mock_services, k=min(3, len(mock_services)))
        
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


class LLMClient:
    """Unified API client wrapper that translates prompts and tool calls between Anthropic, Gemini and OpenAI."""
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


class SingaporeParentsAgent:
    """Orchestrates the LLM agent model reasoning loop and maps LLM function calls to local services."""
    def __init__(self, llm_client: LLMClient, weather_svc: WeatherService, transit_svc: TransitService, bus_svc: BusArrivalService):
        self.llm = llm_client
        self.weather_svc = weather_svc
        self.transit_svc = transit_svc
        self.bus_svc = bus_svc
        
        self.available_tools = {
            "get_current_weather": self.weather_svc.get_current_weather,
            "get_transit_route": self.transit_svc.get_transit_route,
            "get_bus_arrival": self.bus_svc.get_bus_arrival
        }

        self.tools_schema = [
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
                    "name": "get_transit_route",
                    "description": "Get transit directions (MRT/bus trains) between popular locations or stations in Singapore.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "origin": {
                                "type": "string",
                                "description": "The starting location name or latitude,longitude coordinates (e.g. 'Changi Airport', '1.2989,103.8462')."
                            },
                            "destination": {
                                "type": "string",
                                "description": "The destination location name (e.g., 'Marina Bay Sands', 'East Coast Park')."
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

    def run(self, user_message, chat_history=None, chat_id=None, user_locations=None):
        if not self.llm.is_configured():
            return (
                f"⚠️ LLM provider '{self.llm.provider}' is not configured properly. "
                "Please check that you have defined the appropriate API key in your .env file."
            )
            
        if chat_history is None:
            chat_history = []
            
        system_prompt = (
            "You are a helpful, professional travel and family outing agent for Singapore. "
            "You can check current weather, provide MRT/bus routes between popular points, "
            "and query bus arrival times at bus stops. "
            "If the user asks about travelling somewhere, check if your transit route tool has directions. "
            "Keep responses engaging, concise, and structured with clear markdown formatting. "
            "Suggest next steps or other transport queries where appropriate."
        )
        
        # Inject user's shared location if available
        if chat_id and user_locations and chat_id in user_locations:
            loc = user_locations[chat_id]
            system_prompt += (
                f"\n\nCURRENT USER LOCATION: The user has shared their live location: "
                f"Latitude {loc['latitude']}, Longitude {loc['longitude']}. "
                f"If they ask for directions or travel options from 'here', 'my location', 'where I am', etc., "
                f"you MUST pass their exact coordinates '{loc['latitude']},{loc['longitude']}' as the 'origin' parameter "
                f"to the routing tool get_transit_route."
            )
        else:
            system_prompt += (
                "\n\nCURRENT USER LOCATION: The user's current location is unknown. "
                "If they ask for directions from 'here' or 'my location', you must politely ask them "
                "to share their location using the Telegram attachment button so you can help them."
            )
            
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(chat_history[-10:])
        messages.append({"role": "user", "content": user_message})
        
        # Execution loop
        for _ in range(5):
            try:
                response = self.llm.chat_completion(messages=messages, tools=self.tools_schema)
            except Exception as e:
                logger.error(f"LLM API Call Failed: {e}")
                return f"❌ Sorry, I encountered an error communicating with the AI service: {str(e)}"
                
            res_msg = response.choices[0].message
            res_dict = {"role": "assistant", "content": res_msg.content}
            
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
                    
                    if func_name in self.available_tools:
                        tool_func = self.available_tools[func_name]
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
                return res_msg.content
                
        return "⚠️ I couldn't complete the query reasoning loop in a reasonable number of steps. Please try again with a simpler question."


class TelegramBotApp:
    """Manages the Telegram Bot webhook events, message routing, session state, and lifecycle."""
    def __init__(self, token: str, agent: SingaporeParentsAgent):
        self.bot = telebot.TeleBot(token) if token else None
        self.agent = agent
        self.chat_histories = {}
        self.user_locations = {}

        if self.bot:
            self._register_handlers()

    def _register_handlers(self):
        @self.bot.message_handler(commands=['start', 'help'])
        def send_welcome(message):
            welcome_text = (
                "🇸🇬 *Welcome to the Singapore Parents Bot!* 🇸🇬\n\n"
                "I'm here to help you plan your family trips in Singapore! You can ask me:\n"
                "🌤️ *Weather:* \"What's the weather like right now?\"\n"
                "🚇 *Directions:* \"How do I get from Changi Airport to City Hall?\" or \"How do I get to East Coast Park from here?\"\n"
                "🚌 *Bus Arrivals:* \"When is the next bus arriving at stop 01112?\"\n\n"
                "📍 *Tip:* You can share your current location using the attachment button at any time, and I'll use it to give you directions from \"here\"!"
            )
            self.bot.reply_to(message, welcome_text, parse_mode="Markdown")

        @self.bot.message_handler(content_types=['location'])
        def handle_location(message):
            chat_id = message.chat.id
            lat = message.location.latitude
            lon = message.location.longitude
            self.user_locations[chat_id] = {"latitude": lat, "longitude": lon}
            
            nearest = self.agent.transit_svc.get_nearest_landmark(lat, lon)
            reply_text = (
                f"📍 *Location Saved!*\n\n"
                f"I see you are near *{nearest}*. "
                f"Now you can ask me things like: \"How do I get to East Coast Park from here?\""
            )
            self.bot.reply_to(message, reply_text, parse_mode="Markdown", reply_markup=telebot.types.ReplyKeyboardRemove())

        @self.bot.message_handler(func=lambda message: True)
        def handle_user_message(message):
            chat_id = message.chat.id
            user_text = message.text
            
            self.bot.send_chat_action(chat_id, 'typing')
            
            if chat_id not in self.chat_histories:
                self.chat_histories[chat_id] = []
                
            history = self.chat_histories[chat_id]
            
            agent_reply = self.agent.run(
                user_text, 
                history, 
                chat_id=chat_id, 
                user_locations=self.user_locations
            )
            
            history.append({"role": "user", "content": user_text})
            history.append({"role": "assistant", "content": agent_reply})
            
            if len(history) > 10:
                self.chat_histories[chat_id] = history[-10:]
                
            reply_markup = None
            prompt_keywords = ["share your location", "provide your location", "send your location", "send me your location", "where you are"]
            if any(kw in agent_reply.lower() for kw in prompt_keywords):
                reply_markup = telebot.types.ReplyKeyboardMarkup(one_time_keyboard=True, resize_keyboard=True)
                reply_markup.add(telebot.types.KeyboardButton("📍 Share Location", request_location=True))
                
            try:
                self.bot.reply_to(message, agent_reply, parse_mode="Markdown", reply_markup=reply_markup)
            except Exception as e:
                logger.warning(f"Failed to send with Markdown, trying plain text: {e}")
                self.bot.reply_to(message, agent_reply, reply_markup=reply_markup)

    def run(self):
        if not self.bot:
            print("❌ Error: TELEGRAM_BOT_TOKEN and LLM API credentials must be set in your .env file!")
            return
            
        print("🇸🇬 Singapore Travel Agent Telegram Bot is starting up...")
        print("Listening for messages... Press Ctrl+C to stop.")
        try:
            self.bot.infinity_polling()
        except KeyboardInterrupt:
            print("\nStopping bot...")


# --- Main Entry Point ---

if __name__ == "__main__":
    telegram_token = os.getenv("TELEGRAM_BOT_TOKEN")
    lta_key = os.getenv("LTA_ACCOUNT_KEY")
    
    llm_client = LLMClient()
    
    if not telegram_token or not llm_client.is_configured():
        print("❌ Error: TELEGRAM_BOT_TOKEN and LLM API credentials must be set in your .env file!")
    else:
        # Instantiate services
        onemap_auth = OneMapAuth()
        weather_svc = WeatherService()
        transit_svc = TransitService(onemap_auth)
        bus_svc = BusArrivalService(lta_key)
        
        # Instantiate agent orchestrator
        agent = SingaporeParentsAgent(llm_client, weather_svc, transit_svc, bus_svc)
        
        # Start bot application
        app = TelegramBotApp(telegram_token, agent)
        app.run()
