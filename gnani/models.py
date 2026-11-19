"""Published Prisma model coverage and Timbre voice catalogs."""

STT_MODELS = {"prisma-v3.0": "Prisma v3.0", "prisma-v2.5": "Prisma v2.5"}
TTS_MODELS = {"timbre-v2.5": "Timbre v2.5", "timbre-v3.0": "Timbre v3.0"}

INDIC = frozenset(f"{code}-IN" for code in "bn en gu hi kn ml mr pa ta te".split())
STT_LEGACY_LANGUAGES = INDIC
TTS_LANGUAGES = INDIC | {"auto", "hi-en"}

_V25_BY_LANGUAGE = {
    "hi-IN": "Nalini Bhavna Yashvi Urmila Jwala Chitra Ambuja Deepak Roopesh Vikrant Hemraj Jalaj Omkar",
    "en-IN": "Kaveri Trupti Devika Pranav Shlok Girish",
    "hi-en": "Poorvi",
    "ta-IN": "Asmita Trisha Brinda Vedika Noopur",
    "te-IN": "Suhana Lehara Lavanya Yukti Varuni",
    "kn-IN": "Saanvi Kavin",
    "ml-IN": "Reshma Riyaan",
    "mr-IN": "Zahira Ishaan",
    "bn-IN": "Kirra Dhruva",
    "gu-IN": "Falak Veera",
    "pa-IN": "Mehuli Zayan",
}
VOICES_V25 = {
    voice: language for language, voices in _V25_BY_LANGUAGE.items() for voice in voices.split()
}
VOICES_V30 = tuple(
    """Shivani Anitasingh Arko Avinash Chetana Devansh Harshad Kalpana Lakshmi Lakshmidevi
    Mahendra Manjunath Neelam Nitin Prakash Priyasharm Rahulmehta Rajeev Rajni Rakesh Rekha
    Riyaa Santosh Shruthi Shubham Sneha Sohini Sunaina Swathi Tarika Vasudha Vikramdasu Vinod Yashraj""".split()
)
VOICES = {"timbre-v2.5": tuple(VOICES_V25), "timbre-v3.0": VOICES_V30}


def setup_schema(name: str) -> dict:
    return {
        "name": name,
        "badge": "paid",
        "tag": "Gnani AI speech",
        "env_vars": [{"key": "GNANI_API_KEY", "prompt": "Gnani API key", "url": "https://app.gnani.ai"}],
    }
