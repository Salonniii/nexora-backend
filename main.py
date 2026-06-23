from fastapi import FastAPI
from pydantic import BaseModel
from google import genai
import os
import json
import httpx
import re
from dotenv import load_dotenv
from typing import Optional
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()
print("API KEY:", os.getenv("GEMINI_API_KEY"))

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

profile_cache = {}
roadmap_cache = {}
platform_cache = {}

# ✅ Model priority list
# Trying the models most likely to still have real free-tier quota first.
# gemini-1.5-pro removed permanently — it 404s regardless of quota/billing.
MODELS = [
    "gemini-2.5-flash-lite",   # first try — proven working in your logs
    "gemini-2.5-flash",         # second try — also proven working
    "gemini-2.0-flash-lite",    # third try — kept as a backup, not proven but harmless
]
def try_generate(prompt: str) -> str:
    last_error = None
    for model in MODELS:
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt
            )
            print(f"✅ Success with model: {model}")
            return response.text
        except Exception as e:
            print(f"❌ Model {model} failed: {e}")
            last_error = e
    raise last_error

def extract_username(url: str) -> str:
    """Extract username from full URL"""
    if not url:
        return ""
    # Remove trailing slashes
    url = url.rstrip("/")
    # Get last part of URL
    return url.split("/")[-1]


# ==================== PLATFORM DATA FETCHING ====================

async def fetch_repo_readme(client_http: httpx.AsyncClient, username: str, repo_name: str, headers: dict) -> str:
    """Fetch and decode a repo's README.md content (best-effort).

    GitHub's API returns README content as base64. If no README exists,
    or the request fails for any reason, we return an empty string —
    this must never raise, since a missing README is a normal, expected
    case (not an error condition for the overall GitHub fetch).
    """
    try:
        resp = await client_http.get(
            f"https://api.github.com/repos/{username}/{repo_name}/readme",
            headers=headers
        )
        if resp.status_code != 200:
            return ""
        data = resp.json()
        content_b64 = data.get("content", "")
        if not content_b64:
            return ""
        import base64
        decoded = base64.b64decode(content_b64).decode("utf-8", errors="ignore")
        # Strip markdown image/badge syntax and excess whitespace, keep it short.
        decoded = re.sub(r'!\[.*?\]\(.*?\)', '', decoded)   # remove images/badges
        decoded = re.sub(r'<[^>]+>', '', decoded)           # remove raw HTML tags
        decoded = re.sub(r'\n{2,}', '\n', decoded).strip()
        # Cap length so one huge README can't blow up the prompt, but allow
        # enough room for a typical "Features" bullet list to come through.
        return decoded[:900]
    except Exception as e:
        print(f"README fetch error for {repo_name}: {e}")
        return ""

async def fetch_github_data(username: str) -> dict:
    """Fetch real GitHub data using free public API.

    FIX: GitHub's unauthenticated rate limit is only 60 requests/hour PER IP —
    and on Render's free tier, that IP may be shared with other traffic, so it
    can get exhausted fast even with light personal use. This was the actual
    cause of "GitHub data sometimes missing on re-analyze" — it wasn't random,
    the requests were silently hitting GitHub's rate limit and returning a
    non-200 status, which this function correctly treated as "not found."

    If a GITHUB_TOKEN env var is set, we now send it as a Bearer token, which
    raises the limit to 5,000 requests/hour — effectively eliminating this
    failure mode. If no token is set, behavior is unchanged (still works,
    just still subject to the 60/hour limit).
    """
    if not username:
        return {"available": False}
    try:
        github_token = os.getenv("GITHUB_TOKEN")
        headers = {"Accept": "application/vnd.github.v3+json"}
        if github_token:
            headers["Authorization"] = f"Bearer {github_token}"

        async with httpx.AsyncClient(timeout=10) as client_http:
            # User profile
            user_resp = await client_http.get(
                f"https://api.github.com/users/{username}",
                headers=headers
            )
            if user_resp.status_code != 200:
                return {"available": False, "error": f"User not found (status {user_resp.status_code})"}

            user = user_resp.json()

            # Repos
            repos_resp = await client_http.get(
                f"https://api.github.com/users/{username}/repos?per_page=100&sort=updated",
                headers=headers
            )
            repos = repos_resp.json() if repos_resp.status_code == 200 else []
            repos = [r for r in repos if not r.get("fork", False)]

            # Count languages
            languages = {}
            for repo in repos[:10]:  # check top 10 repos
                if repo.get("language"):
                    lang = repo["language"]
                    languages[lang] = languages.get(lang, 0) + 1

            top_languages = sorted(languages, key=languages.get, reverse=True)[:5]

            # FIX: repos often have no "description" field set (common for
            # personal/student projects), even when they have a rich README.
            # Fetch README content for ALL repos missing a description, so
            # Gemini gets real context instead of just a bare repo name.
            # No cap — Render's GitHub token gives 5,000 req/hour, so even a
            # student with 10-20 repos won't come close to that limit.
            recent_repos_raw = repos[:8]  # check up to 8 most-recently-updated repos
            enriched_repos = []
            for r in recent_repos_raw:
                description = r.get("description") or ""
                readme_snippet = ""
                if not description.strip():
                    readme_snippet = await fetch_repo_readme(
                        client_http, username, r["name"], headers
                    )
                enriched_repos.append({
                    "name": r["name"],
                    "description": description,
                    "language": r.get("language", ""),
                    "stars": r.get("stargazers_count", 0),
                    "updated": r.get("updated_at", "")[:10],
                    "readme_snippet": readme_snippet,
                })

            return {
                "available": True,
                "username": username,
                "name": user.get("name", username),
              "public_repos": len(repos),
                "followers": user.get("followers", 0),
                "following": user.get("following", 0),
                "bio": user.get("bio", ""),
                "top_languages": top_languages,
                "recent_repos": enriched_repos,
                "total_stars": sum(r.get("stargazers_count", 0) for r in repos),
            }
    except Exception as e:
        print(f"GitHub fetch error: {e}")
        return {"available": False, "error": str(e)}


async def fetch_leetcode_data(username: str) -> dict:
    """Fetch real LeetCode data using unofficial GraphQL API"""
    if not username:
        return {"available": False}
    try:
        query = """
        {
          matchedUser(username: "%s") {
            username
            submitStats: submitStatsGlobal {
              acSubmissionNum {
                difficulty
                count
                submissions
              }
            }
            profile {
              ranking
              reputation
              starRating
            }
            tagProblemCounts {
              advanced {
                tagName
                problemsSolved
              }
              intermediate {
                tagName
                problemsSolved
              }
              fundamental {
                tagName
                problemsSolved
              }
            }
          }
        }
        """ % username

        async with httpx.AsyncClient(timeout=15) as client_http:
            resp = await client_http.post(
                "https://leetcode.com/graphql",
                json={"query": query},
                headers={
                    "Content-Type": "application/json",
                    "Referer": "https://leetcode.com"
                }
            )

            if resp.status_code != 200:
                return {"available": False, "error": f"Status {resp.status_code}"}

            data = resp.json()
            user = data.get("data", {}).get("matchedUser")

            if not user:
                return {"available": False, "error": "User not found"}

            stats = user.get("submitStats", {}).get("acSubmissionNum", [])
            solved_map = {s["difficulty"]: s["count"] for s in stats}

            # Get topic tags
            tag_counts = user.get("tagProblemCounts", {})
            all_tags = (
                tag_counts.get("advanced", []) +
                tag_counts.get("intermediate", []) +
                tag_counts.get("fundamental", [])
            )
            # Sort by problems solved
            sorted_tags = sorted(all_tags, key=lambda x: x.get("problemsSolved", 0), reverse=True)
            strong_topics = [t["tagName"] for t in sorted_tags[:5] if t.get("problemsSolved", 0) > 0]
            weak_topics = [t["tagName"] for t in sorted_tags[-5:] if t.get("problemsSolved", 0) == 0]

            return {
                "available": True,
                "username": username,
                "total_solved": solved_map.get("All", 0),
                "easy_solved": solved_map.get("Easy", 0),
                "medium_solved": solved_map.get("Medium", 0),
                "hard_solved": solved_map.get("Hard", 0),
                "ranking": user.get("profile", {}).get("ranking", 0),
                "strong_topics": strong_topics,
                "weak_topics": weak_topics,
            }

    except Exception as e:
        print(f"LeetCode fetch error: {e}")
        return {"available": False, "error": str(e)}


async def fetch_gfg_data(username: str) -> dict:
    """Fetch real GFG data using unofficial API"""
    if not username:
        return {"available": False}
    try:
        async with httpx.AsyncClient(timeout=15) as client_http:
            resp = await client_http.get(
                f"https://geeks-for-geeks-stats-api.vercel.app/?userName={username}",
                headers={"Accept": "application/json"}
            )

            if resp.status_code != 200:
                return {"available": False, "error": f"Status {resp.status_code}"}

            data = resp.json()

            if data.get("status") == "error" or not data:
                return {"available": False, "error": "User not found"}

            return {
                "available": True,
                "username": username,
                "total_solved": data.get("totalProblemsSolved", 0),
                "coding_score": data.get("codingScore", 0),
                "monthly_score": data.get("monthlyScore", 0),
                "school": data.get("School", 0),
                "basic": data.get("Basic", 0),
                "easy": data.get("Easy", 0),
                "medium": data.get("Medium", 0),
                "hard": data.get("Hard", 0),
                "institute_rank": data.get("instituteRank", "N/A"),
                "streak": data.get("currentStreak", 0),
                "max_streak": data.get("maxStreak", 0),
            }

    except Exception as e:
        print(f"GFG fetch error: {e}")
        return {"available": False, "error": str(e)}


# ==================== MODELS ====================

class Profile(BaseModel):
    full_name: str
    college: Optional[str] = None
    goal: Optional[str] = None
    github: Optional[str] = None
    linkedin: Optional[str] = None
    gfg: Optional[str] = None
    leetcode: Optional[str] = None

class PlatformRequest(BaseModel):
    github: Optional[str] = None
    leetcode: Optional[str] = None
    gfg: Optional[str] = None

class RoadmapRequest(BaseModel):
    full_name: str
    goal: str
    dream_company: str
    study_hours: int
    branch: str
    year: str
    skills: list[str]
    projects: list[str]
    weak_topics: list[str]
    strong_topics: list[str]

class SmartAnalysisRequest(BaseModel):
    full_name: str
    college: Optional[str] = None
    goal: Optional[str] = None
    dream_company: Optional[str] = None
    github: Optional[str] = None
    leetcode: Optional[str] = None
    gfg: Optional[str] = None


# ==================== ENDPOINTS ====================

@app.get("/")
def home():
    return {"message": "Nexora AI Backend Running 🦊"}


@app.post("/fetch-platform-data", response_model=None)
async def fetch_platform_data(data: PlatformRequest):
    """Fetch real data from GitHub, LeetCode, GFG"""

    github_username = extract_username(data.github or "")
    leetcode_username = extract_username(data.leetcode or "")
    gfg_username = extract_username(data.gfg or "")

    print(f"Fetching: GitHub={github_username}, LC={leetcode_username}, GFG={gfg_username}")

    github_data = await fetch_github_data(github_username) if github_username else {"available": False}
    leetcode_data = await fetch_leetcode_data(leetcode_username) if leetcode_username else {"available": False}
    gfg_data = await fetch_gfg_data(gfg_username) if gfg_username else {"available": False}

    return {
        "github": github_data,
        "leetcode": leetcode_data,
        "gfg": gfg_data,
    }


@app.post("/smart-analyze", response_model=None)
async def smart_analyze(data: SmartAnalysisRequest):
    """
    Fetches REAL data from platforms then uses AI to analyze.
    This is the main powerful endpoint.
    """

    github_username = extract_username(data.github or "")
    leetcode_username = extract_username(data.leetcode or "")
    gfg_username = extract_username(data.gfg or "")

    # Fetch real platform data
    github = await fetch_github_data(github_username) if github_username else {"available": False}
    leetcode = await fetch_leetcode_data(leetcode_username) if leetcode_username else {"available": False}
    gfg = await fetch_gfg_data(gfg_username) if gfg_username else {"available": False}

    # Build context for AI
    platform_context = ""

    if github.get("available"):
        # Build a per-repo breakdown that includes README content when the
        # repo had no description set — this is what actually fixes Gemini
        # seeing "blank" projects that in reality have real content (just in
        # the README rather than the short description field).
        repo_lines = []
        for r in github['recent_repos']:
            line = f"  - {r['name']} ({r['language'] or 'unknown language'})"
            if r.get('description'):
                line += f": {r['description']}"
            elif r.get('readme_snippet'):
                line += f": [from README] {r['readme_snippet']}"
            else:
                line += ": (no description or README available)"
            repo_lines.append(line)

        platform_context += f"""
GITHUB (Real Data):
- Public Repos: {github['public_repos']}
- Total Stars: {github['total_stars']}
- Top Languages: {', '.join(github['top_languages'])}
- Followers: {github['followers']}
- Recent Projects (with real descriptions/README content where available):
{chr(10).join(repo_lines)}
"""

    if leetcode.get("available"):
        platform_context += f"""
LEETCODE (Real Data):
- Total Solved: {leetcode['total_solved']}
- Easy: {leetcode['easy_solved']}, Medium: {leetcode['medium_solved']}, Hard: {leetcode['hard_solved']}
- Strong Topics: {', '.join(leetcode['strong_topics'])}
- Weak/Unsolved Topics: {', '.join(leetcode['weak_topics'])}
- Global Ranking: {leetcode['ranking']}
"""

    if gfg.get("available"):
        platform_context += f"""
GFG (Real Data):
- Total Problems Solved: {gfg['total_solved']}
- Coding Score: {gfg['coding_score']}
- Current Streak: {gfg['streak']} days
- Max Streak: {gfg['max_streak']} days
- Easy: {gfg['easy']}, Medium: {gfg['medium']}, Hard: {gfg['hard']}
- Institute Rank: {gfg['institute_rank']}
"""

    if not platform_context:
        platform_context = "No platform data available - analyze based on profile only."

    print(f"🔍 GitHub available: {github.get('available')}")
    if github.get('available'):
        print(f"🔍 GitHub repos seen by AI: {github.get('public_repos')}")
    print(f"🔍 Full platform_context sent to Gemini:\n{platform_context}")

    prompt = f"""
You are Nova, an AI career coach for STUDENTS preparing for campus placements.
Your audience is undergraduate students, not industry professionals — judge
their numbers against realistic student benchmarks, not senior-engineer standards.

Analyze this student's REAL coding profile data:

Student: {data.full_name}
College: {data.college}
Goal: {data.goal}
Dream Company: {data.dream_company}

{platform_context}

Calibration guide (for a student, NOT a working professional):
- GitHub: 3-5 public repos = reasonable starting point. 6-15 repos = solid, active.
  16+ = strong. Having 0 repos is the actual weak case — do not call a student
  "weak" on GitHub if they have several repos with real project names.
- LeetCode: 50+ solved = engaged. 150+ = strong. 300+ = excellent for a student.
- Always state the ACTUAL NUMBER you were given before judging it
  (e.g. "You have 8 public repositories, which is a solid foundation").

Based on this REAL data, provide:

1. CAREER SCORE (out of 100) - be realistic based on actual numbers, calibrated
   to a student level as described above
2. STRENGTHS (3 specific points based on real data — cite the actual numbers)
3. WEAKNESSES (3 specific points - what topics are missing, what's low —
   only call something "weak" if it is genuinely low for a student, e.g.
   0-2 repos, or 0 hard problems solved, not just below an expert's level)
4. MISSING TOPICS - specific DSA topics they haven't solved yet
5. COMPANY-SPECIFIC QUESTIONS - top 5 questions/topics {data.dream_company} frequently asks
6. DAILY ACTION PLAN - 3 specific things to do TODAY based on their weak areas
7. RECOMMENDATIONS (3 actionable steps)

Be specific and reference their actual numbers. Don't be generic.
If they haven't solved Hard problems, say so. If their GitHub genuinely has
0-2 repos, mention it — but do not describe a reasonable repo count (3+) as weak.
"""


    try:
        text = try_generate(prompt)

        return {
            "analysis": text,
            "platform_data": {
                "github": github,
                "leetcode": leetcode,
                "gfg": gfg,
            }
        }
    except Exception as e:
        return {
            "error": str(e),
            "platform_data": {
                "github": github,
                "leetcode": leetcode,
                "gfg": gfg,
            }
        }

@app.post("/analyze-profile", response_model=None)
def analyze_profile(profile: Profile):
    cache_key = f"{profile.full_name}_{profile.leetcode}_{profile.github}"
    if cache_key in profile_cache:
        return {"analysis": profile_cache[cache_key], "cached": True}

    prompt = f"""
    Analyze this student profile for placement readiness.

    Name: {profile.full_name}
    College: {profile.college}
    Goal: {profile.goal}
    GitHub: {profile.github}
    LinkedIn: {profile.linkedin}
    GFG: {profile.gfg}
    LeetCode: {profile.leetcode}

    Give response in this format:
    1. Career Score out of 100
    2. Strengths (3 points)
    3. Weaknesses (3 points)
    4. Recommendations (3 points)
    """

    try:
        text = try_generate(prompt)
        profile_cache[cache_key] = text
        return {"analysis": text}
    except Exception as e:
        return {"error": str(e), "message": "All Gemini models failed. Try again later."}


@app.post("/generate-roadmap", response_model=None)
def generate_roadmap(data: RoadmapRequest):
    cache_key = f"{data.full_name}_{data.goal}_{data.dream_company}"
    if cache_key in roadmap_cache:
        print("✅ Returning cached roadmap")
        return roadmap_cache[cache_key]

    prompt = f"""
    Create a personalized 30-60-90 day placement preparation roadmap for this student.

    Name: {data.full_name}
    Goal Role: {data.goal}
    Dream Company: {data.dream_company}
    Branch: {data.branch}
    Daily Study Hours: {data.study_hours}
    Year: {data.year}
    Skills: {", ".join(data.skills)}
    Projects: {", ".join(data.projects)}
    Weak Topics: {", ".join(data.weak_topics)}
    Strong Topics: {", ".join(data.strong_topics)}

    IMPORTANT: This student is NOT a beginner. Focus on advanced placement prep.
    Prioritize weak topics. Prepare for top product companies.

    Return ONLY a valid JSON object, no markdown, no explanation, no extra text.
    Use exactly this structure:
    {{
        "day_30": {{
            "title": "Foundation Month",
            "focus": "one sentence focus",
            "tasks": ["task 1", "task 2", "task 3", "task 4", "task 5"]
        }},
        "day_60": {{
            "title": "Building Month",
            "focus": "one sentence focus",
            "tasks": ["task 1", "task 2", "task 3", "task 4", "task 5"]
        }},
        "day_90": {{
            "title": "Final Push Month",
            "focus": "one sentence focus",
            "tasks": ["task 1", "task 2", "task 3", "task 4", "task 5"]
        }},
        "daily_routine": ["routine item 1", "routine item 2", "routine item 3"],
        "must_know_topics": ["topic 1", "topic 2", "topic 3", "topic 4", "topic 5"],
        "resources": ["resource 1", "resource 2", "resource 3"]
    }}
    """

    try:
        text = try_generate(prompt)
        clean_text = text.strip()

        if "```json" in clean_text:
            clean_text = clean_text.split("```json")[1].split("```")[0].strip()
        elif "```" in clean_text:
            clean_text = clean_text.split("```")[1].split("```")[0].strip()

        roadmap_json = json.loads(clean_text)
        roadmap_cache[cache_key] = roadmap_json
        return roadmap_json

    except json.JSONDecodeError as e:
        return {"error": f"JSON parse failed: {e}"}
    except Exception as e:
        return {"error": str(e), "message": "All Gemini models failed. Try again later."}

class ChatRequest(BaseModel):
    message: str
    full_name: Optional[str] = None
    goal: Optional[str] = None


@app.post("/chat-with-nova")
def chat_with_nova(data: ChatRequest):
    prompt = f"""
You are Nova, AI mentor inside Nexora app.

Student Name: {data.full_name}
Career Goal: {data.goal}

User message:
{data.message}

Rules:
- Reply like a friendly smart mentor
- Give practical career advice
- Keep response short (max 120 words)
- Be motivating but honest
"""

    try:
        response = try_generate(prompt)
        return {"reply": response}
    except Exception as e:
        return {"error": str(e)}