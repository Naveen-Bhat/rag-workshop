"""
Agentic RAG Module - Agent with tools for Course Advisor.
Uses the modern langchain.agents.create_agent API.
"""

import json
from pathlib import Path
from typing import Generator

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI

from .rag_chain import CourseAdvisorRAG

# Load environment variables
load_dotenv()


class CourseAdvisorAgent:
    """Agentic RAG system for Course Advisor."""

    def __init__(
        self,
        data_folder: str = "syllabi",
        model_name: str = "gemini-2.0-flash",
        force_reindex: bool = False
    ):
        """Initialize the agent.

        Args:
            data_folder: Folder under data/ containing markdown files
            model_name: Gemini model to use
            force_reindex: If True, rebuild the vector store from source files
        """
        self.data_folder = data_folder
        self.model_name = model_name

        # Load course data
        self._load_course_data()

        # Initialize RAG for search
        self._rag = CourseAdvisorRAG(data_folder=data_folder, force_reindex=force_reindex)

        # Create tools and agent
        self._tools = self._create_tools()
        self._agent_executor = self._create_agent()

    def _load_course_data(self):
        """Load structured course data from JSON."""
        project_root = Path(__file__).parent.parent
        courses_file = project_root / "data" / "courses.json"

        if courses_file.exists():
            with open(courses_file, "r") as f:
                data = json.load(f)
            self.courses = {c["code"]: c for c in data["courses"]}
            self.prereq_chains = data.get("prerequisite_chains", {})
        else:
            self.courses = {}
            self.prereq_chains = {}

    def _create_tools(self):
        """Create agent tools."""
        rag = self._rag
        courses = self.courses

        @tool
        def search_courses(query: str) -> str:
            """Search for courses by topic, name, or description.
            Use this to find courses related to a topic or keyword like 'AI', 'machine learning', 'neural networks'.
            """
            results = rag.search(query, k=3)
            if not results:
                return "No courses found matching that query."

            output = []
            for r in results:
                output.append(f"[{r['source']}]: {r['content'][:200]}...")
            return "\n\n".join(output)

        @tool
        def get_course_info(course_code: str) -> str:
            """Get detailed information about a specific course by its code.
            Example: get_course_info("CS301")
            """
            code = course_code.upper().strip()
            if code in courses:
                c = courses[code]
                prereqs = ", ".join(c["prerequisites"]) or "None"
                return f"""Course: {c['code']} - {c['name']}
Credits: {c['credits']}
Prerequisites: {prereqs}
Instructor: {c['instructor']}
Semester: {c['semester']}
Level: {c['level']}
Topics: {', '.join(c['topics'])}
Description: {c['description']}"""
            return f"Course {code} not found."

        @tool
        def check_prerequisites(course_code: str) -> str:
            """Check the full prerequisite chain for a course.
            This shows ALL prerequisites needed, including indirect ones.
            Example: check_prerequisites("CS401")
            """
            code = course_code.upper().strip()

            def get_all_prereqs(course, visited=None):
                if visited is None:
                    visited = set()
                if course in visited or course not in courses:
                    return []
                visited.add(course)
                direct = courses[course]["prerequisites"]
                all_prereqs = list(direct)
                for prereq in direct:
                    all_prereqs.extend(get_all_prereqs(prereq, visited))
                return all_prereqs

            if code not in courses:
                return f"Course {code} not found."

            direct = courses[code]["prerequisites"]
            all_prereqs = list(set(get_all_prereqs(code)))

            result = f"Prerequisites for {code} ({courses[code]['name']}):\n"
            result += f"  Direct prerequisites: {', '.join(direct) or 'None'}\n"
            result += f"  All prerequisites (including indirect): {', '.join(sorted(all_prereqs)) or 'None'}"

            return result

        @tool
        def find_learning_path(completed_courses: str, target_course: str) -> str:
            """Find the courses needed to reach a target course, given completed courses.

            Args:
                completed_courses: Comma-separated list of completed course codes (e.g., "CS101, MATH101")
                target_course: The course code you want to take (e.g., "CS401")
            """
            completed = set(c.strip().upper() for c in completed_courses.split(",") if c.strip())
            target = target_course.upper().strip()

            if target not in courses:
                return f"Target course {target} not found."

            def get_all_prereqs(course, visited=None):
                if visited is None:
                    visited = set()
                if course in visited or course not in courses:
                    return []
                visited.add(course)
                result = []
                for prereq in courses[course]["prerequisites"]:
                    result.extend(get_all_prereqs(prereq, visited))
                    result.append(prereq)
                return result

            all_needed = get_all_prereqs(target) + [target]

            seen = set()
            ordered = []
            for course in all_needed:
                if course not in seen:
                    seen.add(course)
                    ordered.append(course)

            remaining = [c for c in ordered if c not in completed]

            if not remaining:
                return f"You've completed all prerequisites! You can take {target} now."

            result = f"Learning path from your current courses to {target}:\n"
            result += f"  Completed: {', '.join(sorted(completed)) or 'None'}\n"
            result += f"  Still needed (in order): {' -> '.join(remaining)}"

            return result

        @tool
        def list_all_courses() -> str:
            """List all available courses with their basic information."""
            if not courses:
                return "No course data available."

            result = "Available courses:\n"
            for code, info in sorted(courses.items()):
                prereqs = ", ".join(info["prerequisites"]) or "None"
                result += f"  {code}: {info['name']} (Level: {info['level']}, Prereqs: {prereqs})\n"
            return result

        return [search_courses, get_course_info, check_prerequisites, find_learning_path, list_all_courses]

    def _create_agent(self):
        """Create the ReAct agent using modern API."""
        llm = ChatGoogleGenerativeAI(
            model=self.model_name,
            temperature=0
        )

        system_prompt = """You are a helpful course advisor for Fictional University.
When students ask about courses or learning paths, ALWAYS use your tools to look up information.
Don't ask clarifying questions - instead, search for relevant courses and provide recommendations.

For questions about AI, machine learning, or research paths:
1. First search for relevant courses using search_courses
2. Then check prerequisites using check_prerequisites
3. Finally, suggest a learning path using find_learning_path

Be proactive and helpful - provide complete answers using your tools."""

        return create_agent(llm, self._tools, system_prompt=system_prompt)

    def ask(self, question: str) -> str:
        """Ask a question and get an answer."""
        result = self._agent_executor.invoke({"messages": [("user", question)]})
        return result["messages"][-1].content

    def stream(self, question: str) -> Generator[dict, None, None]:
        """Stream the agent's reasoning process.

        Yields dicts with keys:
        - 'type': 'thought', 'observation', or 'answer'
        - 'content': the text content
        - 'tool': tool name (for thoughts)
        - 'args': tool arguments (for thoughts)
        """
        for step in self._agent_executor.stream({"messages": [("user", question)]}):
            if "model" in step:
                messages = step["model"].get("messages", [])
                for msg in messages:
                    if hasattr(msg, "tool_calls") and msg.tool_calls:
                        for tool_call in msg.tool_calls:
                            yield {
                                "type": "thought",
                                "tool": tool_call["name"],
                                "args": tool_call["args"],
                                "content": f"Using tool '{tool_call['name']}' with {tool_call['args']}"
                            }
                    elif hasattr(msg, "content") and msg.content:
                        yield {
                            "type": "answer",
                            "content": msg.content
                        }

            if "tools" in step:
                messages = step["tools"].get("messages", [])
                for msg in messages:
                    if hasattr(msg, "content"):
                        yield {
                            "type": "observation",
                            "content": msg.content
                        }


def create_course_agent(data_folder: str = "syllabi", **kwargs) -> CourseAdvisorAgent:
    """Create a CourseAdvisorAgent instance."""
    return CourseAdvisorAgent(data_folder=data_folder, **kwargs)
