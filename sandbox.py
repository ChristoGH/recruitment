import asyncio
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
from bs4 import BeautifulSoup

async def main():
    # 1. Browser config - only use supported parameters
    browser_cfg = BrowserConfig(
        browser_type="firefox",
        headless=False,
        verbose=True,
        viewport_width=1920,
        viewport_height=1080
    )

    # 2. Use a simple configuration without a specific extraction strategy
    run_cfg = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
        word_count_threshold=1  # Very low threshold to capture all content
    )

    url = "https://acora.my.salesforce-sites.com/recruit/frecruit__applyjob?vacancyno=vn1081"
    
    async with AsyncWebCrawler(config=browser_cfg) as crawler:
        print(f"Extracting text from {url}")
        result = await crawler.arun(url=url, config=run_cfg)
        
        if result.success:
            print("Extraction succeeded!")
            print("Cleaned HTML length:", len(result.cleaned_html))
            
            # Use BeautifulSoup to extract text from the cleaned HTML
            soup = BeautifulSoup(result.cleaned_html, 'html.parser')
            
            # Extract text from the body
            text_content = soup.get_text(separator='\n', strip=True)
            
            print("\nExtracted text content:")
            print("-" * 50)
            print(text_content)
            print("-" * 50)
            
            # Save the extracted content to a file
            with open("extracted_text.txt", "w", encoding="utf-8") as f:
                f.write(text_content)
            print("\nSaved extracted text to extracted_text.txt")
        else:
            print("Extraction error:", result.error_message)

if __name__ == "__main__":
    asyncio.run(main())