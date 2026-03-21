import json
import time
import os
from playwright.sync_api import sync_playwright

import os
from dotenv import load_dotenv
from config.logger_config import get_logger
logger = get_logger(__name__)

# Load variables from .env file
load_dotenv()


username = os.getenv("USERNAME1")
password = os.getenv("PASSWORD")
chatbotUrl = os.getenv("CHATBOTURL")
baseurl = os.getenv("BASEURL")
logger.info(chatbotUrl)


CHATBOT_URL = chatbotUrl
#INPUT_FILE = "data/input_questions.json"
INPUT_FILE="generated_testset.json"

OUTPUT_FILE = "data/ragas_dataset.json"
SCREENSHOT_DIR = "screenshots"

os.makedirs(SCREENSHOT_DIR, exist_ok=True)

def generate_ui_answers():
    with sync_playwright() as p:
        logger.info("login started on portal")
        browser = p.chromium.launch(headless=False)
        page = browser.new_page()
        page.goto(baseurl)
        page.wait_for_load_state("networkidle")
        page.click("//button")
        page.wait_for_selector('input[type="email"]', state='visible')
        page.fill('input[type="email"]', username)

        page.click('input[type="submit"]')
        page.wait_for_load_state('networkidle')
        page.fill('input[type="password"]', password)

        page.click('input[type="submit"]')
        page.wait_for_timeout(2000)
        page.wait_for_load_state('networkidle')
        page.click('//input[@type="button"]')
        page.goto(CHATBOT_URL)

        # Load questions
        with open(INPUT_FILE, "r") as f:
            questions = json.load(f)
            logger.info('Document loaded')


        results = []

        for i, qa in enumerate(questions):
            question = qa["user_input"]
            ground_truth = qa["reference"]
            reference_contexts= qa["reference_contexts"]


            try:
                logger.info(f"\n Asking Question: {question}")
                # Start new chat
                page.click("//button[text()='Start new chat']")
                page.wait_for_timeout(2000)
                page.wait_for_load_state('networkidle')

                # Fill and send question
                page.get_by_placeholder("Type your message here...").fill(question)
                page.keyboard.press("Enter")
                page.wait_for_timeout(17000)
                page.wait_for_load_state('networkidle')

                # Wait for answer block to appear
                page.wait_for_selector("(//div[@class='w-full max-w-full break-words'])[2]", timeout=15000)

                # Capture all response blocks
                elements = page.locator("(//div[@class='w-full max-w-full break-words'])[2]")
                count = elements.count()
                content_texts = []

                for j in range(count):
                    try:
                        text = elements.nth(j).inner_text().strip()
                        if text:
                            content_texts.append(text)
                    except:
                        continue

                content_text = "\n".join(content_texts).strip()

                if not content_text:
                    raise Exception("No response text found")

                results.append({
                    "question": question,
                    "ground_truth": ground_truth,
                    "answer": content_text,
                    "reference_contexts":reference_contexts

                })

                logger(f" Collected: {question[:60]} -> {content_text[:100]}...")

            except Exception as e:
                logger.error(f" Failed: {question} -> {e}")
                results.append({
                    "question": question,
                    "ground_truth": ground_truth,
                    "answer": "ERROR: No answer",
                    "contexts": []
                })
                screenshot_path = f"{SCREENSHOT_DIR}/fail_{int(time.time())}.png"
                page.screenshot(path=screenshot_path, full_page=True)
                print(f" Screenshot saved to: {screenshot_path}")

        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"\n All answers saved to: {OUTPUT_FILE}")




if __name__ == "__main__":
    generate_ui_answers()
