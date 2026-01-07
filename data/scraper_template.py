"""
Course Scraper Template - Customize this for your college website.

This template provides a starting point for scraping course information
from your college's website. Modify the scraping logic to match your
college's HTML structure.

Usage:
    1. Update BASE_URL to your college's course catalog URL
    2. Modify the scraping functions to match your site's HTML structure
    3. Run: python scraper_template.py
    4. Course files will be saved to data/my_college/

Requirements:
    pip install requests beautifulsoup4
"""

import os
import re
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup


# ============================================================================
# CONFIGURATION - UPDATE THESE FOR YOUR COLLEGE
# ============================================================================

BASE_URL = "https://your-college.edu/courses"  # Change this!
OUTPUT_DIR = Path(__file__).parent / "my_college"

# Rate limiting to be respectful to the server
REQUEST_DELAY = 1.0  # seconds between requests


# ============================================================================
# SCRAPING FUNCTIONS - CUSTOMIZE THESE
# ============================================================================

def fetch_page(url: str) -> BeautifulSoup:
    """Fetch a page and return BeautifulSoup object."""
    print(f"Fetching: {url}")
    response = requests.get(url, headers={
        "User-Agent": "Mozilla/5.0 (Educational Scraper)"
    })
    response.raise_for_status()
    time.sleep(REQUEST_DELAY)
    return BeautifulSoup(response.text, "html.parser")


def get_course_list(soup: BeautifulSoup) -> list:
    """
    Extract list of courses from the catalog page.

    CUSTOMIZE THIS for your college's HTML structure.

    Returns:
        List of dicts with 'code', 'name', 'url' keys
    """
    courses = []

    # Example: Find all course links
    # Modify the selector to match your site
    for item in soup.select(".course-item"):  # Change selector!
        code = item.select_one(".course-code").text.strip()
        name = item.select_one(".course-name").text.strip()
        url = item.select_one("a")["href"]

        courses.append({
            "code": code,
            "name": name,
            "url": url if url.startswith("http") else BASE_URL + url
        })

    return courses


def parse_course_page(soup: BeautifulSoup) -> dict:
    """
    Extract course details from individual course page.

    CUSTOMIZE THIS for your college's HTML structure.

    Returns:
        Dict with course details
    """
    # Example structure - modify for your site
    return {
        "description": soup.select_one(".course-description").text.strip(),
        "prerequisites": soup.select_one(".prerequisites").text.strip(),
        "credits": soup.select_one(".credits").text.strip(),
        "instructor": soup.select_one(".instructor").text.strip(),
        "topics": [li.text.strip() for li in soup.select(".topics li")],
    }


def format_as_markdown(code: str, name: str, details: dict) -> str:
    """Format course data as markdown."""
    topics_list = "\n".join(f"- {t}" for t in details.get("topics", []))

    return f"""# {code}: {name}

## Course Information
- **Credits:** {details.get('credits', 'N/A')}
- **Instructor:** {details.get('instructor', 'TBA')}
- **Prerequisites:** {details.get('prerequisites', 'None')}

## Description
{details.get('description', 'No description available.')}

## Topics Covered
{topics_list if topics_list else '- TBA'}

---
*Scraped from college catalog*
"""


# ============================================================================
# MAIN SCRAPING LOGIC
# ============================================================================

def scrape_courses():
    """Main function to scrape all courses."""
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Scraping courses from: {BASE_URL}")
    print(f"Saving to: {OUTPUT_DIR}")
    print("-" * 50)

    # Fetch main catalog page
    try:
        soup = fetch_page(BASE_URL)
    except Exception as e:
        print(f"Error fetching catalog: {e}")
        print("\nPlease update BASE_URL and selectors for your college.")
        return

    # Get list of courses
    courses = get_course_list(soup)
    print(f"Found {len(courses)} courses")

    # Scrape each course
    for i, course in enumerate(courses, 1):
        print(f"\n[{i}/{len(courses)}] {course['code']}: {course['name']}")

        try:
            # Fetch course page
            course_soup = fetch_page(course["url"])

            # Parse details
            details = parse_course_page(course_soup)

            # Format as markdown
            content = format_as_markdown(course["code"], course["name"], details)

            # Save to file
            filename = f"{course['code'].replace(' ', '_')}.md"
            filepath = OUTPUT_DIR / filename
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content)

            print(f"  Saved: {filename}")

        except Exception as e:
            print(f"  Error: {e}")

    print("\n" + "=" * 50)
    print(f"Done! Scraped {len(courses)} courses to {OUTPUT_DIR}")


# ============================================================================
# MANUAL DATA ENTRY ALTERNATIVE
# ============================================================================

def create_course_manually():
    """
    Interactive function to create course files manually.
    Use this if scraping is too complex for your college's site.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 50)
    print("Manual Course Entry")
    print("=" * 50)
    print("Enter course information (or 'q' to quit)\n")

    while True:
        code = input("Course code (e.g., CS101): ").strip()
        if code.lower() == 'q':
            break

        name = input("Course name: ").strip()
        credits = input("Credits: ").strip()
        instructor = input("Instructor: ").strip()
        prereqs = input("Prerequisites (comma-separated): ").strip()
        description = input("Description (one paragraph): ").strip()

        print("Topics (enter one per line, empty line to finish):")
        topics = []
        while True:
            topic = input("  - ").strip()
            if not topic:
                break
            topics.append(topic)

        # Create markdown content
        topics_list = "\n".join(f"- {t}" for t in topics)
        content = f"""# {code}: {name}

## Course Information
- **Credits:** {credits or 'N/A'}
- **Instructor:** {instructor or 'TBA'}
- **Prerequisites:** {prereqs or 'None'}

## Description
{description or 'No description provided.'}

## Topics Covered
{topics_list if topics_list else '- TBA'}
"""

        # Save file
        filename = f"{code.replace(' ', '_')}.md"
        filepath = OUTPUT_DIR / filename
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)

        print(f"\nSaved: {filepath}")
        print("-" * 30 + "\n")

    print(f"\nDone! Files saved to {OUTPUT_DIR}")


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    print("Course Data Collection")
    print("=" * 50)
    print("1. Scrape from website (requires customization)")
    print("2. Enter courses manually")
    print()

    choice = input("Choose option (1 or 2): ").strip()

    if choice == "1":
        scrape_courses()
    elif choice == "2":
        create_course_manually()
    else:
        print("Invalid choice. Please run again and enter 1 or 2.")
