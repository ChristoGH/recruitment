import asyncio

from crawl4ai.async_configs import BrowserConfig, CrawlerRunConfig, CacheMode
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig

async def main():
    md_generator = DefaultMarkdownGenerator(
        options={
            "ignore_links": True,
            "escape_html": False,
            "body_width": 80
        }
    )
    browser_config = BrowserConfig(verbose=True)
    run_config = CrawlerRunConfig(
        markdown_generator=md_generator,
        # Content filtering
        word_count_threshold=10,
        excluded_tags=['form', 'header'],
        exclude_external_links=True,

        # Content processing
        process_iframes=True,
        remove_overlay_elements=True,

        # Cache control
        cache_mode=CacheMode.ENABLED  # Use cache if available
    )

    async with AsyncWebCrawler(config=browser_config) as crawler:
        result = await crawler.arun(
            # url="https://www.myjobmag.co.za/job/funeral-agent-nigel-agency-middelburg-avbob-south-africa",
            url="https://www.myjobmag.co.za/job/production-pharmacist-port-elizabeth-aspen-pharma-group-9",
            config=run_config
        )

        if result.success:
            # Print clean content
            print("Content:", result.markdown[:20000])  # First 500 chars

            # Process images
            for image in result.media["images"]:
                print(f"Found image: {image['src']}")

            # Process links
            for link in result.links["internal"]:
                print(f"Internal link: {link['href']}")

        else:
            print(f"Crawl failed: {result.error_message}")

if __name__ == "__main__":
    asyncio.run(main())