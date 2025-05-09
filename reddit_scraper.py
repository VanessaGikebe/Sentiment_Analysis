#!/usr/bin/env python3
# reddit_sentiment_analyzer_updated.py

import os
import json
import time
import logging
import argparse
import requests
import re
import pandas as pd
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException, WebDriverException
from webdriver_manager.chrome import ChromeDriverManager

# Sentiment analysis libraries
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import numpy as np

# Download NLTK resources if needed
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')
    nltk.download('punkt')

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("reddit_sentiment_analyzer.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Define the programming languages to analyze with direct URLs to their subreddits
PROGRAMMING_LANGUAGES = {
    'r': {
        'name': 'R',
        'subreddits': ['rlanguage', 'Rlanguage', 'rprogramming', 'RStudio', 'statistics'],
        'color': '#1F77B4'  # Blue
    },
    'rust': {
        'name': 'Rust',
        'subreddits': ['rust', 'rust_gamedev', 'learnrust'],
        'color': '#D62728'  # Red
    },
    'go': {
        'name': 'Go',
        'subreddits': ['golang', 'learngolang'],
        'color': '#2CA02C'  # Green
    }
}

class RedditSentimentAnalyzer:
    def __init__(self, config=None):
        """Initialize the Reddit sentiment analyzer with configuration."""
        self.config = config or {}
        
        # URLs and targets
        self.languages = self.config.get('LANGUAGES', ['r', 'go', 'rust'])
        
        # Browser settings
        self.user_agent = self.config.get('USER_AGENT', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/113.0.0.0 Safari/537.36')
        self.headless = self.config.get('HEADLESS', 'true').lower() == 'true'
        self.timeout = int(self.config.get('TIMEOUT', 30000)) / 1000  # Convert to seconds
        
        # Scraping options
        self.max_pages = int(self.config.get('MAX_PAGES', 3))
        self.max_posts = int(self.config.get('MAX_POSTS', 20))
        
        # Rate limiting
        self.request_delay = int(self.config.get('REQUEST_DELAY', 3000)) / 1000  # Convert to seconds
        
        # Output settings
        self.output_dir = Path(self.config.get('OUTPUT_DIR', 'output'))
        
        # Create directories
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize webdriver
        self.driver = None
        
        # Collection for all posts
        self.all_posts = {}
        self.visited_urls = set()
        self.post_count = {lang: 0 for lang in self.languages}
        
        # Initialize sentiment analyzer
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        
    def setup_driver(self):
        """Set up the Selenium WebDriver with appropriate options."""
        logger.info("Setting up Chrome WebDriver...")
        
        chrome_options = Options()
        
        if self.headless:
            chrome_options.add_argument("--headless=new")
        
        # Set a realistic user agent
        chrome_options.add_argument(f"user-agent={self.user_agent}")
        
        # Performance settings
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        
        # Anti-bot detection settings
        chrome_options.add_argument("--disable-blink-features=AutomationControlled")
        chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        chrome_options.add_experimental_option("useAutomationExtension", False)
        
        # Set window size to desktop resolution
        chrome_options.add_argument("--window-size=1920,1080")
        
        # Enable JavaScript
        chrome_options.add_argument("--enable-javascript")
        
        # Set up Chrome driver
        service = Service(ChromeDriverManager().install())
        self.driver = webdriver.Chrome(service=service, options=chrome_options)
        
        # Set reasonable timeouts
        self.driver.set_page_load_timeout(self.timeout)
        self.driver.set_script_timeout(self.timeout)
        
        # Mask WebDriver to avoid detection
        self.driver.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {
            "source": """
            Object.defineProperty(navigator, 'webdriver', {
                get: () => undefined
            });
            """
        })
        
        logger.info("WebDriver setup complete")
    
    def analyze_sentiment(self, text):
        """Analyze sentiment of text using VADER sentiment analyzer."""
        if not text:
            return {'compound': 0, 'pos': 0, 'neu': 0, 'neg': 0}
            
        return self.sentiment_analyzer.polarity_scores(text)
    
    def start_analysis(self):
        """Start the scraping and sentiment analysis process for all configured languages."""
        try:
            self.setup_driver()
            
            # Initialize data structure for all posts
            for lang in self.languages:
                if lang in PROGRAMMING_LANGUAGES:
                    self.all_posts[lang] = []
                else:
                    logger.warning(f"Unknown language: {lang}, skipping...")
            
            # Process each language
            for lang in self.languages:
                if lang not in PROGRAMMING_LANGUAGES:
                    continue
                    
                lang_info = PROGRAMMING_LANGUAGES[lang]
                logger.info(f"Starting analysis for {lang_info['name']} programming language")
                
                # Access each subreddit for this language
                for subreddit in lang_info['subreddits']:
                    subreddit_url = f"https://www.reddit.com/r/{subreddit}/hot/"
                    logger.info(f"Processing subreddit: r/{subreddit}")
                    
                    try:
                        self.scrape_subreddit(subreddit_url, lang)
                        
                        # Check if we've reached the maximum posts for this language
                        if self.post_count[lang] >= self.max_posts:
                            logger.info(f"Reached maximum post count ({self.max_posts}) for {lang_info['name']}. Moving to next language.")
                            break
                        
                        # Respect rate limits between different subreddits
                        time.sleep(self.request_delay)
                    except Exception as e:
                        logger.error(f"Error processing subreddit {subreddit}: {e}")
                        continue
                
                # Save language results
                lang_output_path = self.output_dir / f"{lang}_posts.json"
                self._save_to_json(self.all_posts[lang], lang_output_path)
                logger.info(f"Completed analysis for {lang_info['name']}. Total posts scraped: {len(self.all_posts[lang])}")
            
            # Perform sentiment analysis and save results
            if any(len(posts) > 0 for posts in self.all_posts.values()):
                self.analyze_all_results()
                logger.info(f"Scraping and analysis complete!")
            else:
                logger.warning("No posts were scraped for any language!")
            
        except Exception as e:
            logger.error(f"Fatal error during analysis: {e}", exc_info=True)
            
            # Try to save what we have so far
            try:
                emergency_path = self.output_dir / 'emergency-save.json'
                self._save_to_json(self.all_posts, emergency_path)
                logger.info(f"Emergency save of data to: {emergency_path}")
            except Exception as save_error:
                logger.error(f"Failed to create emergency save: {save_error}")
                
        finally:
            # Always close the driver
            if self.driver:
                self.driver.quit()
                logger.info("WebDriver closed")
    
    def scrape_subreddit(self, start_url, language):
        """
        Find posts on a subreddit and extract their content.
        """
        current_url = start_url
        page_num = 1
        
        while current_url and page_num <= self.max_pages and self.post_count[language] < self.max_posts:
            if current_url in self.visited_urls:
                logger.info(f"Already visited: {current_url}, skipping...")
                break
            
            logger.info(f"Page {page_num}/{self.max_pages}: {current_url}")
            self.visited_urls.add(current_url)
            
            try:
                # Navigate to the page
                self.driver.get(current_url)
                
                # Wait for content to load - try multiple selectors
                logger.info("Waiting for posts to load...")
                post_found = False
                
                # Try multiple post selectors
                post_selectors = [
                    "div[data-testid='post-container']",
                    "div.thing",
                    "div.Post",
                    "shreddit-post",
                    "[data-click-id='body']"
                ]
                
                for selector in post_selectors:
                    try:
                        WebDriverWait(self.driver, 15).until(
                            EC.presence_of_element_located((By.CSS_SELECTOR, selector))
                        )
                        logger.info(f"Found posts with selector: {selector}")
                        post_found = True
                        break
                    except TimeoutException:
                        continue
                
                if not post_found:
                    logger.warning("No posts found on this page. Trying JavaScript detection...")
                    
                    # Try JavaScript to detect posts
                    try:
                        has_posts = self.driver.execute_script("""
                            return Boolean(
                                document.querySelector('div[data-testid="post-container"]') || 
                                document.querySelector('div.thing') ||
                                document.querySelector('div.Post') ||
                                document.querySelector('shreddit-post') ||
                                document.querySelector('[data-click-id="body"]')
                            );
                        """)
                        
                        if has_posts:
                            logger.info("Found posts using JavaScript detection")
                            post_found = True
                        else:
                            logger.warning("No posts detected even with JavaScript")
                    except Exception as js_error:
                        logger.error(f"JavaScript detection error: {js_error}")
                
                if not post_found:
                    # Last resort - look for any content that might be posts
                    try:
                        # Look for titles, which are often in h3 elements
                        h3_elements = self.driver.find_elements(By.TAG_NAME, "h3")
                        if len(h3_elements) > 3:  # If we have several h3 elements, they might be post titles
                            logger.info(f"Found {len(h3_elements)} potential post titles")
                            post_found = True
                    except Exception as title_err:
                        logger.error(f"Title detection error: {title_err}")
                
                if not post_found:
                    logger.warning("No posts found on this page. Skipping to next URL...")
                    break
                
                # Scroll to load more content
                logger.info("Scrolling to load more content...")
                for i in range(3):
                    self.driver.execute_script("window.scrollBy(0, window.innerHeight);")
                    time.sleep(1)
                
                # Find all post links - try different methods
                post_links = self._extract_post_links()
                
                if not post_links:
                    logger.warning("No post links found, trying alternate methods...")
                    time.sleep(2)  # Wait a bit more
                    
                    # Try once more with different scrolling
                    self.driver.execute_script("window.scrollTo(0, 0);")  # Back to top
                    time.sleep(1)
                    for i in range(5):  # More scrolling
                        self.driver.execute_script(f"window.scrollBy(0, {300 * (i+1)});")
                        time.sleep(0.7)
                    
                    # Try again
                    post_links = self._extract_post_links()
                
                if not post_links:
                    logger.warning("Still couldn't find post links. Moving to next URL.")
                    break
                
                logger.info(f"Found {len(post_links)} post links")
                
                # Visit each post and extract content
                for index, post_url in enumerate(post_links):
                    if self.post_count[language] >= self.max_posts:
                        logger.info(f"Reached maximum post count ({self.max_posts}) for {language}. Stopping.")
                        break
                    
                    if post_url in self.visited_urls:
                        logger.info(f"Already visited post: {post_url}, skipping...")
                        continue
                    
                    logger.info(f"Processing post {index+1}/{len(post_links)}: {post_url}")
                    
                    try:
                        post_data = self._extract_post_content(post_url, language)
                        if post_data:
                            self.all_posts[language].append(post_data)
                            self.post_count[language] += 1
                            
                            # Save batch every 5 posts
                            if self.post_count[language] % 5 == 0:
                                batch_file = self.output_dir / f"{language}_posts_batch_{self.post_count[language]}.json"
                                self._save_to_json(self.all_posts[language], batch_file)
                        
                        # Respect rate limits between posts
                        time.sleep(self.request_delay)
                    
                    except Exception as e:
                        logger.error(f"Error processing post {post_url}: {e}")
                
                # Look for next page button or load more content
                next_url = self._find_next_page_or_load_more()
                
                if next_url and next_url != current_url:
                    current_url = next_url
                    page_num += 1
                else:
                    logger.info("No more pages found")
                    break
                
            except Exception as e:
                logger.error(f"Error processing page {page_num}: {e}")
                break
    
    def _extract_post_links(self):
        """Extract post links using multiple methods."""
        post_links = []
        
        # Method 1: Standard link selectors
        link_selectors = [
            "a[data-click-id='body']",
            "a.title",
            "a[data-testid='post-title']",
            "shreddit-post a[slot='full-post-link']",
            "div.thing a.title"
        ]
        
        for selector in link_selectors:
            try:
                links = self.driver.find_elements(By.CSS_SELECTOR, selector)
                if links:
                    found_links = [link.get_attribute('href') for link in links if link.get_attribute('href')]
                    valid_links = [link for link in found_links if self._is_valid_post_link(link)]
                    if valid_links:
                        logger.info(f"Found {len(valid_links)} post links with selector: {selector}")
                        post_links.extend(valid_links)
            except Exception as e:
                logger.warning(f"Error finding links with selector {selector}: {e}")
        
        # Method 2: JavaScript extraction
        if not post_links:
            try:
                js_links = self.driver.execute_script("""
                    const links = [];
                    
                    // Function to check if a link is a Reddit post link
                    function isPostLink(href) {
                        return href && href.includes('/comments/');
                    }
                    
                    // Try various selectors
                    document.querySelectorAll('a').forEach(link => {
                        if (isPostLink(link.href)) {
                            links.push(link.href);
                        }
                    });
                    
                    // Try shadow DOM if available
                    document.querySelectorAll('shreddit-post').forEach(post => {
                        if (post.shadowRoot) {
                            post.shadowRoot.querySelectorAll('a').forEach(link => {
                                if (isPostLink(link.href)) {
                                    links.push(link.href);
                                }
                            });
                        }
                    });
                    
                    return [...new Set(links)]; // Remove duplicates
                """)
                
                if js_links:
                    valid_js_links = [link for link in js_links if self._is_valid_post_link(link)]
                    logger.info(f"Found {len(valid_js_links)} post links using JavaScript")
                    post_links.extend(valid_js_links)
            except Exception as e:
                logger.error(f"JavaScript link extraction failed: {e}")
        
        # Remove duplicates while preserving order
        return list(dict.fromkeys(post_links))
    
    def _is_valid_post_link(self, url):
        """Check if a URL is a valid Reddit post link."""
        if not url:
            return False
        
        # Valid post URLs typically contain '/comments/' 
        return '/comments/' in url and 'reddit.com/r/' in url
    
    def _extract_post_id(self, url):
        """Extract a unique post ID from a Reddit post URL."""
        # URLs are typically like: https://www.reddit.com/r/subreddit/comments/post_id/post_title/
        if '/comments/' in url:
            parts = url.split('/comments/')
            if len(parts) > 1:
                post_id = parts[1].split('/')[0]
                return post_id
        
        # Fallback to hash
        return str(hash(url) % 10000)
    
    def _extract_post_content(self, post_url, language):
        """Extract the full content of a Reddit post."""
        if post_url in self.visited_urls:
            logger.info(f"Already visited: {post_url}, skipping...")
            return None
        
        self.visited_urls.add(post_url)
        
        try:
            # Navigate to the post page
            self.driver.get(post_url)
            
            # Wait for the post content to load - try multiple selectors with longer timeout
            logger.info("Waiting for post content to load...")
            content_selectors = [
                "div[data-testid='post-content']", 
                "div[slot='text-body']",
                "[data-click-id='text']",
                "div.md",
                ".sitetable .usertext-body",
                "div[data-test-id='post-content']"
            ]
            
            found_content = False
            for selector in content_selectors:
                try:
                    WebDriverWait(self.driver, 15).until(
                        EC.presence_of_element_located((By.CSS_SELECTOR, selector))
                    )
                    logger.info(f"Found post content with selector: {selector}")
                    found_content = True
                    break
                except TimeoutException:
                    continue
            
            # If content wasn't found, still try to extract what we can
            if not found_content:
                logger.warning("Post content elements not found with standard selectors. Attempting extraction anyway.")
            
            # Initialize post data
            post_id = self._extract_post_id(post_url)
            post_data = {
                "post_id": post_id,
                "post_url": post_url,
                "language": language,
                "scraped_at": datetime.now().isoformat()
            }
            
            # Extract post title
            post_data["title"] = self._extract_post_title(post_url, post_id)
            
            # Extract author
            post_data["author"] = self._extract_post_author()
            
            # Extract subreddit
            post_data["subreddit"] = self._extract_post_subreddit(post_url)
            
            # Extract post date
            post_data["post_date"] = self._extract_post_date()
            
            # Extract post content
            post_data["content"] = self._extract_post_text_content()
            
            # Extract metadata (votes, comments)
            self._extract_post_metadata(post_data)
            
            # Analyze sentiment
            if post_data.get("content"):
                post_data["sentiment"] = self.analyze_sentiment(post_data["content"])
            elif post_data.get("title"):
                post_data["sentiment"] = self.analyze_sentiment(post_data["title"])
            else:
                post_data["sentiment"] = {'compound': 0, 'pos': 0, 'neu': 0, 'neg': 0}
            
            logger.info(f"Successfully extracted post data: {post_data.get('title', 'Untitled')}")
            logger.info(f"Sentiment: {post_data['sentiment']['compound']:.2f} (pos: {post_data['sentiment']['pos']:.2f}, neg: {post_data['sentiment']['neg']:.2f})")
            
            return post_data
            
        except Exception as e:
            logger.error(f"Error extracting post content for {post_url}: {e}", exc_info=True)
            return None
    
    def _extract_post_title(self, post_url, post_id):
        """Extract the post title using multiple methods."""
        try:
            # Method 1: Standard selectors
            title_selectors = ["h1", "h1[slot='title']", ".title a", "[data-testid='post-title']"]
            for selector in title_selectors:
                try:
                    title_element = self.driver.find_element(By.CSS_SELECTOR, selector)
                    title = title_element.text.strip()
                    if title:
                        return title
                except NoSuchElementException:
                    continue
            
            # Method 2: JavaScript extraction
            try:
                title = self.driver.execute_script("""
                    // Try various title selectors
                    const selectors = ['h1', 'h1[slot="title"]', '.title a', '[data-testid="post-title"]'];
                    for (const selector of selectors) {
                        const element = document.querySelector(selector);
                        if (element && element.textContent.trim()) {
                            return element.textContent.trim();
                        }
                    }
                    
                    // Try shadow DOM
                    const postElement = document.querySelector('shreddit-post');
                    if (postElement && postElement.shadowRoot) {
                        const titleElem = postElement.shadowRoot.querySelector('h1, [slot="title"]');
                        if (titleElem) {
                            return titleElem.textContent.trim();
                        }
                    }
                    
                    return null;
                """)
                
                if title:
                    return title
            except Exception:
                pass
            
            # Method 3: Extract from URL as last resort
            url_parts = post_url.split('/')
            if len(url_parts) > 1:
                potential_title = url_parts[-2] if url_parts[-1] == '' else url_parts[-1]
                return potential_title.replace('_', ' ').replace('-', ' ').title()
                
            # Final fallback
            return f"Post {post_id}"
            
        except Exception as e:
            logger.error(f"Error extracting title: {e}")
            return f"Post {post_id}"
    
    def _extract_post_author(self):
        """Extract the post author."""
        try:
            author_selectors = [
                "a[data-testid='post_author']", 
                "a[slot='author']",
                ".author",
                "a.author"
            ]
            for selector in author_selectors:
                try:
                    author_element = self.driver.find_element(By.CSS_SELECTOR, selector)
                    author = author_element.text.strip()
                    if author:
                        return author
                except NoSuchElementException:
                    continue
            
            # JavaScript extraction
            try:
                author = self.driver.execute_script("""
                    const selectors = ['a[data-testid="post_author"]', 'a[slot="author"]', '.author', 'a.author'];
                    for (const selector of selectors) {
                        const element = document.querySelector(selector);
                        if (element && element.textContent.trim()) {
                            return element.textContent.trim();
                        }
                    }
                    
                    // Try shadow DOM
                    const postElement = document.querySelector('shreddit-post');
                    if (postElement && postElement.shadowRoot) {
                        const authorElem = postElement.shadowRoot.querySelector('a[slot="author"]');
                        if (authorElem) {
                            return authorElem.textContent.trim();
                        }
                    }
                    
                    return null;
                """)
                
                if author:
                    return author
            except Exception:
                pass
            
            return "Unknown"
        except Exception as e:
            logger.error(f"Error extracting author: {e}")
            return "Unknown"
    
    def _extract_post_subreddit(self, post_url):
        """Extract the subreddit from the post."""
        try:
            # Method 1: From selectors
            subreddit_selectors = [
                "a[data-testid='subreddit-link']", 
                "a[slot='subreddit-name']",
                ".subreddit",
                "a.subreddit"
            ]
            for selector in subreddit_selectors:
                try:
                    subreddit_element = self.driver.find_element(By.CSS_SELECTOR, selector)
                    subreddit = subreddit_element.text.strip()
                    if subreddit:
                        return subreddit
                except NoSuchElementException:
                    continue
            
            # Method 2: From URL
            if "/r/" in post_url:
                match = re.search(r"/r/([^/]+)", post_url)
                if match:
                    return f"r/{match.group(1)}"
            
            return "Unknown Subreddit"
        except Exception as e:
            logger.error(f"Error extracting subreddit: {e}")
            return "Unknown Subreddit"
    
    def _extract_post_date(self):
        """Extract the post date."""
        try:
            date_selectors = [
                "span[data-testid='post_timestamp']", 
                "span[slot='posted-time']",
                "time",
                ".tagline time"
            ]
            for selector in date_selectors:
                try:
                    date_element = self.driver.find_element(By.CSS_SELECTOR, selector)
                    date = date_element.text.strip()
                    if date:
                        return date
                except NoSuchElementException:
                    continue
            
            # JavaScript extraction
            try:
                date = self.driver.execute_script("""
                    const selectors = ['span[data-testid="post_timestamp"]', 'span[slot="posted-time"]', 'time', '.tagline time'];
                    for (const selector of selectors) {
                        const element = document.querySelector(selector);
                        if (element && element.textContent.trim()) {
                            return element.textContent.trim();
                        }
                    }
                    
                    // Try shadow DOM
                    const postElement = document.querySelector('shreddit-post');
                    if (postElement && postElement.shadowRoot) {
                        const timeElem = postElement.shadowRoot.querySelector('span[slot="posted-time"]');
                        if (timeElem) {
                            return timeElem.textContent.trim();
                        }
                    }
                    
                    return null;
                """)
                
                if date:
                    return date
            except Exception:
                pass
            
            return "Unknown Date"
        except Exception as e:
            logger.error(f"Error extracting date: {e}")
            return "Unknown Date"
    
    def _extract_post_text_content(self):
        """Extract the post text content."""
        try:
            # Method 1: Standard selectors
            content_selectors = [
                "div[data-test-id='post-content']", 
                "div[slot='text-body']", 
                "[data-click-id='text']",
                "div.md",
                ".sitetable .usertext-body"
            ]
            
            for content_selector in content_selectors:
                try:
                    content_element = self.driver.find_element(By.CSS_SELECTOR, content_selector)
                    content = content_element.text.strip()
                    if content:
                        return content
                except NoSuchElementException:
                    continue
            
            # Method 2: JavaScript extraction
            try:
                content = self.driver.execute_script("""
                    // Try various content selectors
                    const selectors = ['div[data-test-id="post-content"]', 'div[slot="text-body"]', '[data-click-id="text"]', 'div.md', '.sitetable .usertext-body'];
                    for (const selector of selectors) {
                        const element = document.querySelector(selector);
                        if (element && element.textContent.trim()) {
                            return element.textContent.trim();
                        }
                    }
                    
                    // Try shadow DOM
                    const postElement = document.querySelector('shreddit-post');
                    if (postElement && postElement.shadowRoot) {
                        const contentElem = postElement.shadowRoot.querySelector('div[slot="text-body"], [data-click-id="text"]');
                        if (contentElem) {
                            return contentElem.textContent.trim();
                        }
                    }
                    
                    return null;
                """)
                
                if content:
                    return content
            except Exception as e:
                logger.error(f"JavaScript content extraction failed: {e}")
            
            # Method 3: Paragraph extraction
            try:
                paragraphs = self.driver.find_elements(By.TAG_NAME, "p")
                # Filter out navigation, header, and footer paragraphs
                content_paragraphs = []
                for p in paragraphs:
                    # Skip small paragraphs that are likely UI elements
                    if len(p.text.strip()) < 20:
                        continue
                        
                    # Skip paragraphs in common UI areas
                    if self._is_in_ui_area(p):
                        continue
                        
                    content_paragraphs.append(p.text.strip())
                
                if content_paragraphs:
                    return "\n\n".join(content_paragraphs)
            except Exception as e:
                logger.error(f"Paragraph extraction failed: {e}")
            
            # No content found
            return ""
        except Exception as e:
            logger.error(f"Error extracting content: {e}")
            return ""
    
    def _extract_post_metadata(self, post_data):
        """Extract additional metadata like votes and comments count."""
        try:
            # Extract votes
            vote_selectors = ["div[id^='vote-arrows-']", "faceplate-number[slot='upvote-count']", ".score.unvoted", ".score"]
            for selector in vote_selectors:
                try:
                    votes_element = self.driver.find_element(By.CSS_SELECTOR, selector)
                    post_data["votes"] = votes_element.text.strip()
                    break
                except NoSuchElementException:
                    continue
            
            # JavaScript extraction for votes if needed
            if "votes" not in post_data:
                try:
                    votes = self.driver.execute_script("""
                        const selectors = ["div[id^='vote-arrows-']", "faceplate-number[slot='upvote-count']", ".score.unvoted", ".score"];
                        for (const selector of selectors) {
                            const element = document.querySelector(selector);
                            if (element && element.textContent.trim()) {
                                return element.textContent.trim();
                            }
                        }
                        
                        // Try shadow DOM
                        const postElement = document.querySelector('shreddit-post');
                        if (postElement && postElement.shadowRoot) {
                            const votesElem = postElement.shadowRoot.querySelector("faceplate-number[slot='upvote-count']");
                            if (votesElem) {
                                return votesElem.textContent.trim();
                            }
                        }
                        
                        return null;
                    """)
                    
                    if votes:
                        post_data["votes"] = votes
                except Exception:
                    pass
            
            # Extract comments count
            comment_selectors = ["span[data-click-id='comments']", "span[slot='comment-count']", ".comments", "a.comments"]
            for selector in comment_selectors:
                try:
                    comments_element = self.driver.find_element(By.CSS_SELECTOR, selector)
                    post_data["comments_count"] = comments_element.text.strip()
                    break
                except NoSuchElementException:
                    continue
                    
            # JavaScript extraction for comments if needed
            if "comments_count" not in post_data:
                try:
                    comments = self.driver.execute_script("""
                        const selectors = ["span[data-click-id='comments']", "span[slot='comment-count']", ".comments", "a.comments"];
                        for (const selector of selectors) {
                            const element = document.querySelector(selector);
                            if (element && element.textContent.trim()) {
                                return element.textContent.trim();
                            }
                        }
                        
                        // Try shadow DOM
                        const postElement = document.querySelector('shreddit-post');
                        if (postElement && postElement.shadowRoot) {
                            const commentsElem = postElement.shadowRoot.querySelector("span[slot='comment-count']");
                            if (commentsElem) {
                                return commentsElem.textContent.trim();
                            }
                        }
                        
                        return null;
                    """)
                    
                    if comments:
                        post_data["comments_count"] = comments
                except Exception:
                    pass
        except Exception as e:
            logger.error(f"Error extracting metadata: {e}")
    
    def _is_in_ui_area(self, element):
        """Check if an element is part of the UI rather than content."""
        try:
            # Check parent elements
            parent = element
            for _ in range(5):  # Check up to 5 levels up
                if not parent:
                    break
                    
                parent_class = parent.get_attribute("class") or ""
                parent_id = parent.get_attribute("id") or ""
                
                # Common UI area classes and IDs
                ui_indicators = [
                    "header", "footer", "sidebar", "navigation", "nav", "menu", 
                    "comment", "toolbar", "banner", "ad", "widget", "modal", 
                    "popup", "overlay", "tooltip", "community-widget"
                ]
                
                for indicator in ui_indicators:
                    if indicator in parent_class.lower() or indicator in parent_id.lower():
                        return True
                
                # Get parent
                parent = parent.find_element(By.XPATH, "..")
        except:
            pass
            
        return False
        
    def _find_next_page_or_load_more(self):
        """Find the next page button or load more content button."""
        try:
            # Method 1: Look for standard next page buttons
            next_button_selectors = [
                "button[aria-label='Next']",
                "a.next-button",
                "span.next-button a",
                "a[rel='next']",
                ".nav-buttons .next-button a",
                "a.next"
            ]
            
            for selector in next_button_selectors:
                try:
                    next_button = self.driver.find_element(By.CSS_SELECTOR, selector)
                    
                    # Skip if disabled
                    if next_button.get_attribute("disabled") == "true":
                        continue
                    
                    logger.info(f"Found next button with selector: {selector}")
                    
                    # If it's a link, get the href
                    if next_button.tag_name.lower() == "a":
                        return next_button.get_attribute("href")
                    
                    # If it's a button, click it and get the new URL
                    next_button.click()
                    time.sleep(2)  # Wait for navigation
                    return self.driver.current_url
                except NoSuchElementException:
                    continue
                except Exception as e:
                    logger.warning(f"Error with next button: {e}")
            
            # Method 2: Look for "Load more" buttons
            load_more_selectors = [
                "button:contains('Load more')",
                "button.MoreCommentsLink",
                "a:contains('load more')",
                "button[data-click-id='load-more']",
                "button.button:contains('More')"
            ]
            
            for selector in load_more_selectors:
                try:
                    # For :contains() pseudo-selector, need to use JavaScript
                    if ":contains" in selector:
                        text = selector.split(":contains('")[1].split("')")[0]
                        script = f"""
                            const buttons = Array.from(document.querySelectorAll('button, a'));
                            return buttons.find(btn => btn.textContent.includes('{text}'));
                        """
                        load_more = self.driver.execute_script(script)
                    else:
                        load_more = self.driver.find_element(By.CSS_SELECTOR, selector)
                        
                    if load_more:
                        logger.info(f"Found load more button")
                        load_more.click()
                        time.sleep(3)  # Wait for content to load
                        return self.driver.current_url
                except Exception:
                    continue
            
            # Method 3: Check for infinite scroll - scroll down and see if URL changes or new content loads
            try:
                old_content = len(self.driver.find_elements(By.CSS_SELECTOR, "div[data-testid='post-container'], shreddit-post, div.thing"))
                self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(3)  # Wait for possible content to load
                new_content = len(self.driver.find_elements(By.CSS_SELECTOR, "div[data-testid='post-container'], shreddit-post, div.thing"))
                
                if new_content > old_content:
                    logger.info(f"Detected infinite scroll - new content loaded ({old_content} -> {new_content})")
                    return self.driver.current_url
            except Exception as e:
                logger.error(f"Error checking for infinite scroll: {e}")
            
            # Method 4: Check URL for page number and increment
            current_url = self.driver.current_url
            page_match = re.search(r'page=(\d+)', current_url)
            if page_match:
                current_page = int(page_match.group(1))
                next_page = current_page + 1
                next_url = re.sub(r'page=\d+', f'page={next_page}', current_url)
                return next_url
            elif '?' in current_url:
                return f"{current_url}&page=2"
            else:
                return f"{current_url}?page=2"
                
        except Exception as e:
            logger.error(f"Error finding next page: {e}")
            return None
    
    def _save_to_json(self, data, filepath):
        """Save data to a JSON file."""
        try:
            filepath.parent.mkdir(exist_ok=True, parents=True)
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            logger.info(f"Data saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving to JSON: {e}")
    
    def analyze_all_results(self):
        """Analyze all results and create JSON output files."""
        try:
            logger.info("Analyzing sentiment data for all languages...")
            
            # Prepare DataFrame for sentiment analysis
            sentiment_data = []
            
            for lang in self.languages:
                if lang not in self.all_posts or not self.all_posts[lang]:
                    continue
                
                for post in self.all_posts[lang]:
                    if 'sentiment' in post:
                        # Extract language info
                        lang_name = PROGRAMMING_LANGUAGES.get(lang, {}).get('name', lang)
                        
                        # Add entry
                        sentiment_data.append({
                            'language': lang_name,
                            'post_id': post.get('post_id', ''),
                            'title': post.get('title', ''),
                            'subreddit': post.get('subreddit', ''),
                            'compound': post['sentiment'].get('compound', 0),
                            'positive': post['sentiment'].get('pos', 0),
                            'neutral': post['sentiment'].get('neu', 0),
                            'negative': post['sentiment'].get('neg', 0),
                        })
            
            if not sentiment_data:
                logger.warning("No sentiment data available for analysis!")
                return
            
            # Convert to DataFrame
            df = pd.DataFrame(sentiment_data)
            
            # Save full sentiment data
            sentiment_csv_path = self.output_dir / "sentiment_analysis.csv"
            df.to_csv(sentiment_csv_path, index=False)
            logger.info(f"Sentiment data saved to {sentiment_csv_path}")
            
            # Create JSON outputs
            self._create_json_outputs(df)
            
        except Exception as e:
            logger.error(f"Error analyzing results: {e}", exc_info=True)
    
    def _create_json_outputs(self, df):
        """Create JSON output of sentiment analysis results."""
        try:
            logger.info("Creating JSON outputs...")
            
            # Calculate average sentiment by language
            avg_sentiment = df.groupby('language')['compound'].mean().to_dict()
            
            # Calculate positive and negative components
            pos_components = df.groupby('language')['positive'].mean().to_dict()
            neg_components = df.groupby('language')['negative'].mean().to_dict()
            
            # Prepare summary data
            summary_data = {
                'average_sentiment': avg_sentiment,
                'positive_components': pos_components,
                'negative_components': neg_components,
                'post_counts': df.groupby('language').size().to_dict(),
                'generated_at': datetime.now().isoformat()
            }
            
            # Save summary JSON
            summary_path = self.output_dir / "sentiment_summary.json"
            self._save_to_json(summary_data, summary_path)
            logger.info(f"Summary JSON saved to {summary_path}")
            
            # Create detailed language JSON files
            for language in df['language'].unique():
                language_df = df[df['language'] == language]
                
                # Calculate statistics
                stats = {
                    'language': language,
                    'average_sentiment': language_df['compound'].mean(),
                    'median_sentiment': language_df['compound'].median(),
                    'std_deviation': language_df['compound'].std(),
                    'positive_component': language_df['positive'].mean(),
                    'negative_component': language_df['negative'].mean(),
                    'neutral_component': language_df['neutral'].mean(),
                    'post_count': len(language_df),
                    'sentiment_distribution': {
                        'very_positive': len(language_df[language_df['compound'] > 0.5]),
                        'positive': len(language_df[(language_df['compound'] <= 0.5) & (language_df['compound'] > 0.05)]),
                        'neutral': len(language_df[(language_df['compound'] >= -0.05) & (language_df['compound'] <= 0.05)]),
                        'negative': len(language_df[(language_df['compound'] >= -0.5) & (language_df['compound'] < -0.05)]),
                        'very_negative': len(language_df[language_df['compound'] < -0.5])
                    }
                }
                
                # Save language stats
                lang_key = language.lower()
                lang_stats_path = self.output_dir / f"{lang_key}_sentiment_stats.json"
                self._save_to_json(stats, lang_stats_path)
                logger.info(f"{language} stats saved to {lang_stats_path}")
            
            # Create complete results file with detailed posts and sentiment
            complete_data = []
            for lang in self.languages:
                if lang not in self.all_posts or not self.all_posts[lang]:
                    continue
                    
                lang_name = PROGRAMMING_LANGUAGES.get(lang, {}).get('name', lang)
                lang_data = {
                    'language': lang_name,
                    'posts': self.all_posts[lang]
                }
                
                complete_data.append(lang_data)
            
            complete_path = self.output_dir / "complete_sentiment_analysis.json"
            self._save_to_json(complete_data, complete_path)
            logger.info(f"Complete analysis saved to {complete_path}")
            
        except Exception as e:
            logger.error(f"Error creating JSON outputs: {e}", exc_info=True)


def load_config_from_env():
    """Load configuration from environment variables."""
    config = {
        # Languages to analyze
        'LANGUAGES': os.getenv('LANGUAGES', 'r,go,rust'),
        
        # Browser settings
        'USER_AGENT': os.getenv('USER_AGENT', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/113.0.0.0 Safari/537.36'),
        'HEADLESS': os.getenv('HEADLESS', 'true'),
        'TIMEOUT': os.getenv('TIMEOUT', '30000'),
        
        # Scraping options
        'MAX_PAGES': os.getenv('MAX_PAGES', '3'),
        'MAX_POSTS': os.getenv('MAX_POSTS', '20'),
        
        # Rate limiting
        'REQUEST_DELAY': os.getenv('REQUEST_DELAY', '3000'),
        
        # Output settings
        'OUTPUT_DIR': os.getenv('OUTPUT_DIR', 'output'),
    }
    
    # Convert comma-separated string to list
    if 'LANGUAGES' in config:
        config['LANGUAGES'] = [lang.strip() for lang in config['LANGUAGES'].split(',') if lang.strip()]
    
    return config


def main():
    """Main function to run the Reddit sentiment analyzer."""
    parser = argparse.ArgumentParser(description='Reddit Programming Language Sentiment Analyzer')
    parser.add_argument('--languages', type=str, help='Comma-separated list of programming languages to analyze (r,go,rust)')
    parser.add_argument('--max-pages', type=int, help='Maximum pages to scrape per source')
    parser.add_argument('--max-posts', type=int, help='Maximum total posts to scrape per language')
    parser.add_argument('--headless', action='store_true', help='Run in headless mode')
    parser.add_argument('--output-dir', type=str, help='Directory to save output files')
    args = parser.parse_args()
    
    # Load config from .env file
    config = load_config_from_env()
    
    # Override with command line arguments
    if args.languages:
        config['LANGUAGES'] = [lang.strip() for lang in args.languages.split(',') if lang.strip()]
    
    if args.max_pages:
        config['MAX_PAGES'] = str(args.max_pages)
    
    if args.max_posts:
        config['MAX_POSTS'] = str(args.max_posts)
    
    if args.headless:
        config['HEADLESS'] = 'true'
    
    if args.output_dir:
        config['OUTPUT_DIR'] = args.output_dir
    
    # Initialize and run the analyzer
    analyzer = RedditSentimentAnalyzer(config)
    analyzer.start_analysis()


if __name__ == "__main__":
    main()