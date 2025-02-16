import time
import json
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# Base page
BASE_URL = "https://dorar.net/feqhia"
SAVE_PATH = "data3.json"

# List of target categories
TARGET_CATEGORIES = ["كتابُ الزَّكاةِ" , "كتابُ الصَّوم"

]

# Track expanded elements to avoid duplicate clicks
expanded_sections = set()

# Headers for requests
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36"
    )
}


def fetch_lecture_text(url):
    """
    Fetches lecture text using requests instead of Selenium.
    """
    try:
        resp = requests.get(url, headers=HEADERS, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        # Try different selectors in case the structure changes
        possible_selectors = ["div.w-100.mt-4", "div.card-text", "div.content"]
        for selector in possible_selectors:
            main_text_div = soup.select_one(selector)
            if main_text_div:
                return main_text_div.get_text(strip=True)

        return "Lecture content not found."

    except requests.exceptions.RequestException as e:
        return f"Error fetching lecture: {str(e)}"


def expand_and_collect_links(driver, li_element, path_so_far):
    """
    Recursively expands sections and collects lecture links.
    """
    collected = {}
    num_lectures = 0

    # Get section name
    try:
        section_name = li_element.find_element(By.CSS_SELECTOR, "a[style='cursor: pointer;']").text.strip()
    except:
        section_name = "Unnamed Section"

    # Skip re-expanding already expanded sections
    if section_name in expanded_sections:
        return collected, num_lectures
    expanded_sections.add(section_name)

    # Click to expand if it's not expanded already
    try:
        clickable = li_element.find_element(By.CSS_SELECTOR, "a[style='cursor: pointer;']")
        driver.execute_script("arguments[0].click();", clickable)
        time.sleep(1)  # Small delay for UI update
    except:
        pass  # It's likely already expanded

    # Extract lecture links
    links_in_li = li_element.find_elements(By.CSS_SELECTOR, "a[href*='/feqhia/']")
    for link_el in links_in_li:
        href = link_el.get_attribute("href")
        title = link_el.text.strip()

        if href and title:
            num_lectures += 1
            lecture_content = fetch_lecture_text(href)
            section = path_so_far[-1] if path_so_far else "General"
            collected.setdefault(section, []).append({
                "lecture_title": title,
                "lecture_url": href,
                "content": lecture_content,
                "path": " > ".join(path_so_far)
            })

    # Recursively process sub-sections
    sub_li_elements = li_element.find_elements(By.CSS_SELECTOR, ":scope > ul > li.mtree-node")
    for sub_li in sub_li_elements:
        try:
            sub_title = sub_li.find_element(By.CSS_SELECTOR, "a[style='cursor: pointer;']").text.strip()
        except:
            sub_title = sub_li.text.strip().split("\n")[0]

        new_path = path_so_far + [sub_title]
        deeper_data, num_sub = expand_and_collect_links(driver, sub_li, new_path)
        num_lectures += num_sub

        for key, value in deeper_data.items():
            collected.setdefault(key, []).extend(value)

    return collected, num_lectures


def scrape_filtered_category(driver, category_element, category_name):
    """
    Scrapes a single category and saves data as JSON.
    """
    print(f"🔍 Processing category: {category_name}")
    category_data, num_lectures = expand_and_collect_links(driver, category_element, [category_name])

    # Save category data
    save_data_as_json(category_data)

    print(f"✅ Finished {category_name}: {num_lectures} lectures extracted.")
    return num_lectures


def scrape_filtered_categories():
    """
    Scrapes only the selected categories and saves results.
    """
    driver = webdriver.Chrome()
    driver.get(BASE_URL)
    wait = WebDriverWait(driver, 15)

    # Wait for the tree to load
    wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "ul#mtree")))

    total_lectures = 0

    # Get all top-level categories
    top_level_lis = driver.find_elements(By.CSS_SELECTOR, "ul#mtree > li.mtree-node")

    for li_el in top_level_lis:
        li_text = li_el.text.strip()

        # Match only the selected categories
        if any(cat in li_text for cat in TARGET_CATEGORIES):
            total_lectures += scrape_filtered_category(driver, li_el, li_text)

    driver.quit()
    print(f"📊 Total lectures scraped: {total_lectures}")


def save_data_as_json(data, save_path=SAVE_PATH):
    """
    Saves the lecture data as JSON.
    """
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    start_time = time.time()
    scrape_filtered_categories()
    elapsed_time = time.time() - start_time
    print(f"✅ Scraping complete! Execution time: {elapsed_time:.2f} seconds")
